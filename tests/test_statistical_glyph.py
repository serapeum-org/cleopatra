import doctest

import matplotlib
import numpy as np
import pytest

matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

import cleopatra.statistical_glyph as statistical_glyph_module
from cleopatra.statistical_glyph import StatisticalGlyph


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


def test_module_doctests_execute():
    """Run the module's docstring examples so they are exercised in CI.

    Pytest is not configured with ``--doctest-modules``, so docstring examples in
    ``src/`` would otherwise never run. This test executes them for
    ``cleopatra.statistical_glyph`` (including the fig/ax composition example) and
    fails if any example's output no longer matches.
    """
    try:
        results = doctest.testmod(statistical_glyph_module, verbose=False)
    finally:
        plt.close("all")
    assert results.failed == 0, f"{results.failed} doctest example(s) failed in statistical_glyph"
    assert results.attempted > 0, (
        "no doctest examples were collected from statistical_glyph; the module's docstring "
        "examples may have been moved or removed, silently dropping this coverage"
    )


def test_histogram_does_not_call_plt_show(monkeypatch):
    """Test that `StatisticalGlyph.histogram()` does not force an interactive display.

    Args:
        monkeypatch: Pytest fixture used to replace ``matplotlib.pyplot.show``.

    Test scenario:
        Patch ``plt.show`` with a counter. After ``histogram()`` the counter must be 0,
        and the method must still return its (fig, ax, hist) triple. Pins the contract that
        histogram() returns its figure for the caller to display rather than showing it.
    """
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    np.random.seed(1)
    x = 4 + np.random.normal(0, 1.5, 200)
    fig, ax, hist = StatisticalGlyph(x).histogram()
    assert calls == [], f"histogram() should not call plt.show(); was called {len(calls)} time(s)"
    assert isinstance(fig, Figure), f"histogram() should return a Figure, got {type(fig)}"
    assert isinstance(hist, dict), f"histogram() should return a dict of results, got {type(hist)}"


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


class TestHistogramFigAxInjection:
    """Tests for the ``fig``/``ax`` composition parameters of ``StatisticalGlyph``."""

    @staticmethod
    def _data():
        """Return a fixed 1D sample so bin counts are deterministic.

        Returns:
            np.ndarray: 200 normally distributed values seeded for reproducibility.
        """
        np.random.seed(1)
        return 4 + np.random.normal(0, 1.5, 200)

    def teardown_method(self):
        """Close all figures after each test to avoid leaking matplotlib state."""
        plt.close("all")

    def test_ax_supplied_draws_into_that_ax_and_infers_figure(self):
        """Test that a supplied ``ax`` is reused and its parent figure is inferred.

        Test scenario:
            Construct with ``ax=`` only. ``histogram()`` must return that exact axes,
            return the axes' parent figure (no new figure), and draw the bars on it.
        """
        fig0, ax0 = plt.subplots()
        stat = StatisticalGlyph(self._data(), ax=ax0)
        fig, ax, hist = stat.histogram()
        assert ax is ax0, "histogram() should draw into the supplied axes"
        assert fig is fig0, f"figure should be inferred from the axes, got {fig}"
        assert len(ax0.patches) > 0, "expected histogram bars on the supplied axes"

    def test_ax_and_fig_supplied_both_reused(self):
        """Test that explicitly supplied ``fig`` and ``ax`` are both returned unchanged.

        Test scenario:
            Construct with both ``fig=`` and ``ax=``. ``histogram()`` must return those
            exact objects rather than creating new ones.
        """
        fig0, ax0 = plt.subplots()
        stat = StatisticalGlyph(self._data(), fig=fig0, ax=ax0)
        fig, ax, _ = stat.histogram()
        assert fig is fig0, "supplied figure should be returned unchanged"
        assert ax is ax0, "supplied axes should be returned unchanged"

    def test_fig_supplied_without_ax_creates_axes_on_that_figure(self):
        """Test the M1 fix: a ``fig`` without ``ax`` is honoured, not discarded.

        Test scenario:
            Construct with ``fig=`` only (no axes). ``histogram()`` must return that exact
            figure and create the axes *on* it, instead of silently building a new figure.
        """
        fig0 = plt.figure()
        assert fig0.axes == [], "precondition: the supplied figure starts with no axes"
        stat = StatisticalGlyph(self._data(), fig=fig0)
        fig, ax, _ = stat.histogram()
        assert fig is fig0, "histogram() should reuse the supplied figure"
        assert ax in fig0.axes, "the created axes should belong to the supplied figure"
        assert len(ax.patches) > 0, "expected histogram bars on the created axes"

    def test_fig_supplied_without_ax_raises_when_figure_already_has_axes(self):
        """Test that fig-only mode rejects a figure that already contains axes.

        Test scenario:
            Pass a ``fig`` that already has a populated 1x2 layout but no ``ax``.
            ``histogram()`` must raise ``ValueError`` directing the caller to pass ``ax``
            explicitly, rather than overlaying a full-figure axes on the existing panels.
        """
        fig0, _ = plt.subplots(1, 2)
        stat = StatisticalGlyph(self._data(), fig=fig0)
        with pytest.raises(ValueError, match=r"already contains axes") as exc:
            stat.histogram()
        assert "ax=" in str(exc.value), f"error should point the caller to `ax=`, got: {exc.value}"

    def test_no_fig_no_ax_creates_new_figure_and_axes(self):
        """Test that omitting both ``fig`` and ``ax`` creates fresh objects.

        Test scenario:
            Construct with neither argument. ``histogram()`` must return a brand-new
            Figure/Axes pair that is unrelated to any pre-existing figure.
        """
        other_fig, other_ax = plt.subplots()
        stat = StatisticalGlyph(self._data())
        fig, ax, _ = stat.histogram()
        assert isinstance(fig, Figure), f"histogram() should return a Figure, got {type(fig)}"
        assert isinstance(ax, Axes), f"histogram() should return an Axes, got {type(ax)}"
        assert fig is not other_fig, "a new figure should have been created"
        assert ax is not other_ax, "a new axes should have been created"

    def test_injected_axes_receives_styling(self):
        """Test that axis labels/ticks are applied to the injected axes, not pyplot state.

        Test scenario:
            With a supplied axes that is *not* the pyplot current axes, the configured
            ``xlabel``/``ylabel`` must land on that axes (guards the ``plt.*`` -> ``ax.*``
            change that makes injection correct).
        """
        fig0, ax0 = plt.subplots()
        plt.figure()  # make a different figure the pyplot "current" one
        stat = StatisticalGlyph(self._data(), ax=ax0, xlabel="X values", ylabel="Counts")
        _, ax, _ = stat.histogram()
        assert ax.get_xlabel() == "X values", f"xlabel not applied to injected axes: {ax.get_xlabel()!r}"
        assert ax.get_ylabel() == "Counts", f"ylabel not applied to injected axes: {ax.get_ylabel()!r}"


class TestStatisticalGlyphValidationAndState:
    """Tests for the ``values`` setter and ``histogram()`` validation guards."""

    def teardown_method(self):
        """Close all figures after each test to avoid leaking matplotlib state."""
        plt.close("all")

    def test_values_setter_replaces_stored_values(self):
        """Test that assigning ``stat.values`` swaps the underlying data.

        Test scenario:
            Construct with one 1D sample, assign a differently-shaped array via the
            ``values`` setter, and confirm the property returns the new array.
        """
        np.random.seed(1)
        stat = StatisticalGlyph(np.random.normal(0, 1, 100))
        new_values = np.random.normal(5, 2, 50)
        stat.values = new_values
        assert stat.values is new_values, "the values setter should store the assigned array"
        assert stat.values.shape == (50,), f"expected shape (50,), got {stat.values.shape}"

    def test_histogram_invalid_kwarg_raises(self):
        """Test that an unknown ``histogram()`` keyword raises ``ValueError``.

        Test scenario:
            Pass a keyword that is not a recognised option; ``histogram()`` must raise
            ``ValueError`` naming the offending argument.
        """
        np.random.seed(1)
        stat = StatisticalGlyph(np.random.normal(0, 1, 100))
        with pytest.raises(ValueError, match=r"not correct") as exc:
            stat.histogram(not_a_real_option=123)
        assert "not_a_real_option" in str(exc.value), f"error should name the bad kwarg: {exc.value}"

    def test_histogram_color_count_mismatch_raises(self):
        """Test that 2D data with too few colors raises ``ValueError``.

        Test scenario:
            Build a 3-column dataset but leave the default single-color option, so the
            number of colors (1) does not match the number of samples (3). ``histogram()``
            must raise ``ValueError`` explaining the mismatch.
        """
        np.random.seed(1)
        data_2d = np.random.normal(0, 1, (100, 3))
        stat = StatisticalGlyph(data_2d)
        with pytest.raises(ValueError, match=r"number of colors") as exc:
            stat.histogram()
        assert "samples:3" in str(exc.value), f"error should report the sample count: {exc.value}"


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

    def test_positions_via_kwargs_keep_ticks_aligned(self):
        """Forwarded `positions` keep tick locations under the boxes (L1 fix).

        Test scenario:
            Passing positions through **kwargs places the boxes at those
            x positions, and the tick locations follow them (rather than
            staying at the default 1..n), so labels stay under the boxes.
        """
        np.random.seed(1)
        stat = StatisticalGlyph(
            np.random.normal(0, 1, (20, 3)), color=["r", "g", "b"]
        )
        _, ax, bp = stat.boxplot(positions=[10, 20, 30])
        box_x = [round(float(line.get_xdata().mean())) for line in bp["medians"]]
        assert box_x == [10, 20, 30], f"Boxes should sit at positions, got {box_x}"
        assert list(ax.get_xticks()) == [10, 20, 30], (
            f"Ticks should follow positions, got {list(ax.get_xticks())}"
        )
        assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2", "3"], (
            "Default labels should still be 1-based indices"
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

    def test_return_annotation_resolves(self):
        """stripes' return type hint resolves to BarContainer (N1 fix).

        Test scenario:
            typing.get_type_hints succeeds (the annotation no longer
            references an unimported submodule) and the return type is
            matplotlib's BarContainer.
        """
        import typing

        hints = typing.get_type_hints(StatisticalGlyph.stripes)
        assert typing.get_args(hints["return"]) == (Figure, Axes, BarContainer), (
            f"Unexpected resolved return hint: {hints['return']}"
        )
