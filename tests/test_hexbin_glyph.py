"""Tests for cleopatra.glyphs.stats.hexbin_glyph.HexbinGlyph — issue #353.

Covers construction/validation, the count vs `reduce` aggregate, `min_count`
dropping sparse bins, `gridsize` as an int and a pair, `evaluate()` agreeing
with the drawn `PolyCollection`, the colorbar contract, classification, and
the colour-scale pipeline.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, LogNorm, to_rgba

from cleopatra.glyphs.stats.hexbin_glyph import HexbinGlyph
from cleopatra.styling.params import Classify, Contour
from cleopatra.styling.scaling import ColorScaling


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


@pytest.fixture()
def cloud():
    """A reproducible 2-D point cloud with a per-point value array.

    Returns:
        tuple: `(x, y, values)`, each a length-400 float array.
    """
    rng = np.random.default_rng(42)
    x = rng.normal(size=400)
    y = rng.normal(size=400)
    values = x + y
    return x, y, values


class TestHexbinConstruction:
    """Tests for HexbinGlyph.__init__ validation and stored state."""

    def test_stores_coordinates_as_float_arrays(self, cloud):
        """x and y are stored as float ndarrays."""
        x, y, _ = cloud
        glyph = HexbinGlyph(x, y)
        assert glyph.x.dtype == float and glyph.y.dtype == float, (
            "coords should be float"
        )
        assert glyph.values is None, "values should default to None (counts)"

    def test_mismatched_xy_lengths_raise(self):
        """x and y of different lengths raise ValueError."""
        with pytest.raises(ValueError, match="same shape"):
            HexbinGlyph(np.arange(5.0), np.arange(4.0))

    def test_mismatched_values_length_raises(self):
        """A values array not matching x/y raises ValueError."""
        x = np.arange(5.0)
        with pytest.raises(ValueError, match="values must match"):
            HexbinGlyph(x, x, np.arange(4.0))

    def test_non_1d_input_raises(self):
        """2-D x/y inputs raise ValueError."""
        grid = np.zeros((3, 3))
        with pytest.raises(ValueError, match="must be 1-D"):
            HexbinGlyph(grid, grid)

    def test_empty_input_raises(self):
        """An empty point cloud raises ValueError."""
        with pytest.raises(ValueError, match="at least one point"):
            HexbinGlyph(np.array([]), np.array([]))


class TestHexbinEvaluate:
    """Tests for HexbinGlyph.evaluate (binning without rendering)."""

    def test_returns_aligned_centres_and_aggregate(self, cloud):
        """evaluate returns bin-centre x/y and a per-bin aggregate of equal length."""
        x, y, _ = cloud
        cx, cy, agg = HexbinGlyph(x, y, gridsize=8).evaluate()
        assert cx.shape == cy.shape == agg.shape, "centres and aggregate must align"
        assert cx.ndim == 1 and cx.size > 0, "there should be at least one bin"

    def test_counts_sum_to_point_total(self, cloud):
        """With no values, the per-bin counts sum to the number of points."""
        x, y, _ = cloud
        _, _, agg = HexbinGlyph(x, y, gridsize=10).evaluate()
        assert int(np.nansum(agg)) == x.size, "counts should total the point count"

    def test_agrees_with_drawn_collection(self, cloud):
        """evaluate's aggregate matches the array of the PolyCollection plot draws."""
        x, y, v = cloud
        glyph = HexbinGlyph(x, y, v, gridsize=9, reduce="mean")
        _, _, agg = glyph.evaluate()
        _, _, pc = glyph.plot()
        drawn = np.ma.asarray(pc.get_array()).astype(float).filled(np.nan)
        assert np.allclose(agg, drawn, equal_nan=True), (
            "evaluate must match the drawn array"
        )

    def test_does_not_touch_the_glyph_axes(self, cloud):
        """evaluate renders on a throwaway figure, leaving self.ax unset."""
        x, y, _ = cloud
        glyph = HexbinGlyph(x, y)
        glyph.evaluate()
        assert glyph.ax is None, "evaluate must not create or bind the glyph's axes"

    def test_does_not_leak_a_global_figure(self, cloud):
        """evaluate renders on a bare Figure() and leaks no pyplot-managed figure."""
        x, y, _ = cloud
        before = set(plt.get_fignums())
        HexbinGlyph(x, y).evaluate()
        assert set(plt.get_fignums()) == before, (
            "evaluate must not leak a global figure"
        )


class TestHexbinReduce:
    """Tests for the `reduce` aggregation option."""

    def test_count_of_values_totals_points(self, cloud):
        """reduce='count' with a values array totals the number of points."""
        x, y, v = cloud
        _, _, pc = HexbinGlyph(x, y, v, reduce="count", gridsize=10).plot()
        assert int(np.ma.asarray(pc.get_array()).sum()) == x.size

    def test_sum_equals_mean_times_count(self, cloud):
        """reduce='sum' bins agree with mean*count on the same grid."""
        x, y, v = cloud
        _, _, cnt = HexbinGlyph(x, y, v, reduce="count", gridsize=6).evaluate()
        _, _, mean = HexbinGlyph(x, y, v, reduce="mean", gridsize=6).evaluate()
        _, _, total = HexbinGlyph(x, y, v, reduce="sum", gridsize=6).evaluate()
        assert np.allclose(total, mean * cnt, equal_nan=True), (
            "sum should equal mean*count"
        )

    def test_callable_reduce_is_forwarded(self, cloud):
        """A callable `reduce` is passed straight to hexbin."""
        x, y, v = cloud
        _, _, pc = HexbinGlyph(x, y, v, reduce=np.median, gridsize=6).plot()
        assert isinstance(pc, PolyCollection), "a callable reduce should render"

    def test_unknown_reduce_raises(self, cloud):
        """An unknown `reduce` name raises a clear ValueError."""
        x, y, v = cloud
        with pytest.raises(ValueError, match="reduce must be"):
            HexbinGlyph(x, y, v, reduce="nope").plot()


class TestHexbinGrid:
    """Tests for gridsize and min_count."""

    def test_gridsize_int(self, cloud):
        """An integer gridsize renders."""
        x, y, _ = cloud
        _, _, pc = HexbinGlyph(x, y, gridsize=15).plot()
        assert isinstance(pc, PolyCollection)

    def test_gridsize_pair(self, cloud):
        """A (nx, ny) gridsize pair renders."""
        x, y, _ = cloud
        _, _, pc = HexbinGlyph(x, y, gridsize=(8, 5)).plot()
        assert isinstance(pc, PolyCollection)

    def test_extent_excluding_data_raises_clear_error(self, cloud):
        """An extent that excludes every point raises a clear, actionable error."""
        x, y, v = cloud
        glyph = HexbinGlyph(x, y, v, extent=(100.0, 200.0, 100.0, 200.0))
        with pytest.raises(ValueError, match="no hexagonal bins to draw"):
            glyph.plot()

    def test_min_count_above_densest_bin_raises_clear_error(self, cloud):
        """A min_count above the densest bin drops every cell with a clear error."""
        x, y, _ = cloud
        glyph = HexbinGlyph(x, y, min_count=10_000)
        with pytest.raises(ValueError, match="no hexagonal bins to draw"):
            glyph.plot()

    def test_min_count_drops_sparse_bins(self, cloud):
        """A higher min_count leaves no fewer... i.e. drops sparsely-populated bins."""
        x, y, _ = cloud
        _, _, loose = HexbinGlyph(x, y, gridsize=20, min_count=1).evaluate()
        _, _, strict = HexbinGlyph(x, y, gridsize=20, min_count=5).evaluate()
        assert strict.size < loose.size, "a larger min_count should drop sparse bins"
        assert np.nanmin(strict) >= 5, "surviving bins hold at least min_count points"


class TestHexbinPlot:
    """Tests for the HexbinGlyph.plot rendering contract."""

    def test_returns_fig_ax_collection(self, cloud):
        """plot returns (Figure, Axes, PolyCollection) and stores the collection."""
        x, y, _ = cloud
        glyph = HexbinGlyph(x, y)
        fig, ax, pc = glyph.plot()
        assert fig is glyph.fig and ax is glyph.ax, "returned fig/ax are the glyph's"
        assert isinstance(pc, PolyCollection) and glyph.im is pc, (
            "im holds the collection"
        )

    def test_colorbar_drawn_by_default(self, cloud):
        """A colorbar is attached by default."""
        x, y, _ = cloud
        glyph = HexbinGlyph(x, y)
        glyph.plot()
        assert glyph.cbar is not None, "a colorbar should be drawn by default"

    def test_add_colorbar_false_suppresses(self, cloud):
        """add_colorbar=False suppresses the colorbar."""
        x, y, _ = cloud
        glyph = HexbinGlyph(x, y)
        glyph.plot(add_colorbar=False)
        assert glyph.cbar is None, "add_colorbar=False should suppress the colorbar"

    def test_title_override(self, cloud):
        """A title passed to plot sets the axes title."""
        x, y, _ = cloud
        _, ax, _ = HexbinGlyph(x, y).plot(title="Density")
        assert ax.get_title() == "Density"

    def test_vmin_vmax_honoured(self, cloud):
        """Explicit vmin/vmax reach the collection's norm."""
        x, y, v = cloud
        _, _, pc = HexbinGlyph(x, y, v, vmin=-1.0, vmax=1.0, gridsize=8).plot()
        assert pc.norm.vmin == pytest.approx(-1.0) and pc.norm.vmax == pytest.approx(
            1.0
        )

    def test_draws_on_supplied_axes(self, cloud):
        """plot(ax=...) draws on the caller's axes."""
        x, y, _ = cloud
        fig, ax = plt.subplots()
        _, used, _ = HexbinGlyph(x, y).plot(ax=ax)
        assert used is ax, "the supplied axes should be used"

    def test_replot_without_ax_reuses_bound_axes(self, cloud):
        """A second plot() with no ax reuses the axes bound by the first."""
        x, y, _ = cloud
        glyph = HexbinGlyph(x, y)
        _, ax1, _ = glyph.plot()
        _, ax2, _ = glyph.plot()
        assert ax2 is ax1, "re-plot without ax should reuse the bound axes"

    def test_edge_color_and_line_width_reach_collection(self, cloud):
        """edge_color and line_width options reach the drawn PolyCollection."""
        x, y, _ = cloud
        _, _, pc = HexbinGlyph(x, y, edge_color="red", line_width=1.5).plot()
        assert pc.get_linewidths()[0] == pytest.approx(1.5), "line_width not applied"
        assert tuple(pc.get_edgecolors()[0]) == to_rgba("red"), "edge_color not applied"

    def test_extent_crops_the_binning_window(self):
        """A valid extent bounds the bin centres to that window."""
        rng = np.random.default_rng(7)
        x, y = rng.uniform(0.0, 10.0, 600), rng.uniform(0.0, 10.0, 600)
        cx, cy, _ = HexbinGlyph(
            x, y, gridsize=8, extent=(2.0, 6.0, 3.0, 7.0)
        ).evaluate()
        assert 1.5 <= cx.min() and cx.max() <= 6.5, "extent did not bound x centres"
        assert 2.5 <= cy.min() and cy.max() <= 7.5, "extent did not bound y centres"


class TestHexbinClassify:
    """Tests for the classify / colour-scale group parameters."""

    def test_quantiles_scheme_discretises(self, cloud):
        """classify=Classify(scheme='quantiles') yields a BoundaryNorm and a colorbar."""
        x, y, v = cloud
        glyph = HexbinGlyph(x, y, v, gridsize=10)
        _, _, pc = glyph.plot(classify=Classify(scheme="quantiles", k=4))
        assert isinstance(pc.norm, BoundaryNorm), (
            "a scheme should produce a BoundaryNorm"
        )
        assert glyph.cbar is not None, "a classified hexbin still draws a colorbar"

    def test_categorical_scheme_rejected(self, cloud):
        """A categorical scheme is rejected: a per-bin aggregate is continuous."""
        x, y, _ = cloud
        bad = Classify(scheme="categorical")
        with pytest.raises(ValueError):
            HexbinGlyph(x, y).plot(classify=bad)

    def test_contour_levels_discretise(self, cloud):
        """contour=Contour(levels=n) discretises the colour scale."""
        x, y, _ = cloud
        _, _, pc = HexbinGlyph(x, y, gridsize=10, min_count=1).plot(
            contour=Contour(levels=5)
        )
        assert isinstance(pc.norm, BoundaryNorm), "levels should discretise the scale"

    def test_color_scaling_applied(self, cloud):
        """color=ColorScaling.log() applies a log norm to positive counts."""
        x, y, _ = cloud
        _, _, pc = HexbinGlyph(x, y, gridsize=10, min_count=1).plot(
            color=ColorScaling.log()
        )
        assert isinstance(pc.norm, LogNorm), "ColorScaling.log() should apply a LogNorm"

    def test_scheme_and_k_are_option_keys(self):
        """HexbinGlyph exposes scheme/k as accepted options (pipeline integration)."""
        keys = HexbinGlyph.option_keys()
        assert "scheme" in keys and "k" in keys, "hexbin should accept scheme/k"
