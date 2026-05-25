import doctest

import matplotlib
import numpy as np
import pytest

matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import cleopatra.statistical_glyph as statistical_glyph_module
from cleopatra.statistical_glyph import StatisticalGlyph


def test_module_doctests_execute():
    """Run the module's docstring examples so they are exercised in CI.

    Pytest is not configured with ``--doctest-modules``, so docstring examples in
    ``src/`` would otherwise never run. This test executes them for
    ``cleopatra.statistical_glyph`` (including the fig/ax composition example) and
    fails if any example's output no longer matches.
    """
    results = doctest.testmod(statistical_glyph_module, verbose=False)
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
        assert isinstance(fig, Figure) and isinstance(ax, Axes)
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
