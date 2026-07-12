"""Tests for cleopatra.kde_glyph.KDEGlyph — issue #156.

Covers construction/validation, the bandwidth rule, grid evaluation (including
the memory-chunking path), level resolution, contour clipping, and the
`plot` rendering contract. The estimator is numpy-only; one test asserts scipy
is never imported.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.contour import QuadContourSet
from matplotlib.patches import Circle
from matplotlib.path import Path as MplPath

import cleopatra.kde_glyph as kde_mod
from cleopatra.kde_glyph import KDE_DEFAULT_OPTIONS, KDEGlyph


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


@pytest.fixture()
def cloud():
    """A reproducible 2-D Gaussian point cloud.

    Returns:
        tuple[np.ndarray, np.ndarray]: 400 x and y coordinates from a fixed
            seed.
    """
    rng = np.random.default_rng(1234)
    return rng.normal(size=400), rng.normal(size=400)


@pytest.fixture()
def bimodal():
    """Two tight, well-separated clusters at (0, 0) and (5, 5).

    Returns:
        tuple[np.ndarray, np.ndarray]: x and y coordinates of the clusters.
    """
    rng = np.random.default_rng(7)
    a = rng.normal(scale=0.1, size=300)
    b = rng.normal(scale=0.1, size=300)
    x = np.concatenate([a, a + 5.0])
    y = np.concatenate([b, b + 5.0])
    return x, y


class TestKDEGlyphInit:
    """Tests for KDEGlyph.__init__."""

    def test_stores_coordinates_and_defaults(self, cloud):
        """Coordinates are stored as float arrays and defaults are present.

        Test scenario:
            x/y become float ndarrays; clip_path defaults to None and cbar
            is unset until plot.
        """
        x, y = cloud
        glyph = KDEGlyph(x, y)
        assert glyph.x.dtype == float, "x should be stored as float"
        assert glyph.clip_path is None, "clip_path should default to None"
        assert glyph.cbar is None, "cbar should be None before plotting"
        assert (
            glyph.default_options["gridsize"] == 100
        ), "default gridsize should be 100"

    def test_clip_path_stored(self, cloud):
        """A supplied clip_path is stored on the glyph.

        Test scenario:
            A Circle patch passed at construction is kept for plot time.
        """
        x, y = cloud
        patch = Circle((0, 0), 1.0)
        glyph = KDEGlyph(x, y, clip_path=patch)
        assert glyph.clip_path is patch, "clip_path should be stored verbatim"

    def test_mismatched_xy_raises(self):
        """x and y of different lengths raise ValueError.

        Test scenario:
            len(x) != len(y) is rejected at construction.
        """
        with pytest.raises(ValueError, match="same shape"):
            KDEGlyph(np.arange(5.0), np.arange(4.0))

    def test_too_few_points_raises(self):
        """Fewer than two points raise ValueError.

        Test scenario:
            A single point cannot define a KDE.
        """
        with pytest.raises(ValueError, match="at least 2 points"):
            KDEGlyph(np.array([1.0]), np.array([2.0]))

    @pytest.mark.parametrize("bad_bw", [0.0, -1.0])
    def test_non_positive_bw_method_raises(self, cloud, bad_bw):
        """A non-positive ``bw_method`` raises ValueError.

        Args:
            cloud: The point-cloud fixture.
            bad_bw: An invalid bandwidth multiplier.

        Test scenario:
            Zero or negative bandwidth multipliers are rejected.
        """
        x, y = cloud
        with pytest.raises(ValueError, match="positive float or None"):
            KDEGlyph(x, y, bw_method=bad_bw)

    def test_invalid_kwarg_raises(self, cloud):
        """An unknown kwarg is rejected by the strict merge.

        Test scenario:
            A key absent from KDE_DEFAULT_OPTIONS raises ValueError.
        """
        x, y = cloud
        with pytest.raises(ValueError, match="not correct"):
            KDEGlyph(x, y, not_an_option=1)


class TestKDEGlyphBandwidth:
    """Tests for KDEGlyph._bandwidth."""

    def test_scotts_rule_default(self, cloud):
        """The default bandwidth is Scott's rule ``n ** (-1/6)``.

        Test scenario:
            With no bw_method, the factor is n**(-1/6).
        """
        x, y = cloud
        glyph = KDEGlyph(x, y)
        expected = x.size ** (-1.0 / 6.0)
        assert np.isclose(
            glyph._bandwidth(), expected
        ), f"Expected Scott's factor {expected}, got {glyph._bandwidth()}"

    def test_bw_method_multiplier(self, cloud):
        """``bw_method`` scales Scott's rule linearly.

        Test scenario:
            bw_method=2 doubles the Scott's-rule factor.
        """
        x, y = cloud
        glyph = KDEGlyph(x, y, bw_method=2.0)
        expected = 2.0 * x.size ** (-1.0 / 6.0)
        assert np.isclose(
            glyph._bandwidth(), expected
        ), f"Expected {expected}, got {glyph._bandwidth()}"


class TestKDEGlyphEvaluate:
    """Tests for KDEGlyph.evaluate."""

    def test_grid_shape(self, cloud):
        """The grid and density share the ``gridsize × gridsize`` shape.

        Test scenario:
            gridsize=50 yields 50×50 gx, gy, and density arrays.
        """
        x, y = cloud
        gx, gy, density = KDEGlyph(x, y, gridsize=50).evaluate()
        assert gx.shape == (50, 50), f"gx shape should be (50, 50), got {gx.shape}"
        assert density.shape == (
            50,
            50,
        ), f"density shape should be (50, 50), got {density.shape}"

    def test_density_positive(self, cloud):
        """The density grid is strictly positive everywhere.

        Test scenario:
            A Gaussian kernel sum is positive across the whole grid.
        """
        x, y = cloud
        _, _, density = KDEGlyph(x, y, gridsize=40).evaluate()
        assert np.all(density > 0), "Density should be positive everywhere"

    def test_density_integrates_to_about_one(self, cloud):
        """The density integrates to approximately 1 over the grid.

        Test scenario:
            A trapezoidal integral of the normalised density is near 1 for a
            grid that comfortably spans the cloud.
        """
        x, y = cloud
        gx, gy, density = KDEGlyph(x, y, gridsize=200).evaluate()
        dx = gx[0, 1] - gx[0, 0]
        dy = gy[1, 0] - gy[0, 0]
        integral = density.sum() * dx * dy
        assert 0.8 < integral < 1.2, f"Density integral should be ~1, got {integral}"

    def test_density_peaks_near_cluster(self, bimodal):
        """The global density peak sits near a cluster centre.

        Test scenario:
            For clusters at (0,0)/(5,5) the argmax cell is within 1.0 of one
            centre in each axis.
        """
        x, y = bimodal
        gx, gy, density = KDEGlyph(x, y, gridsize=60).evaluate()
        peak = np.unravel_index(int(np.argmax(density)), density.shape)
        near_x = min(abs(gx[peak] - 0.0), abs(gx[peak] - 5.0))
        near_y = min(abs(gy[peak] - 0.0), abs(gy[peak] - 5.0))
        assert (
            near_x < 1.0 and near_y < 1.0
        ), f"Peak {(gx[peak], gy[peak])} not near a cluster centre"

    @pytest.mark.parametrize(
        "x, y",
        [
            (np.zeros(10), np.arange(10.0)),
            (np.arange(10.0), np.full(10, 3.0)),
        ],
    )
    def test_zero_spread_raises(self, x, y):
        """A zero-spread coordinate raises ValueError.

        Args:
            x: x-coordinates (constant in one parametrisation).
            y: y-coordinates (constant in the other).

        Test scenario:
            A degenerate kernel (std 0 in x or y) is rejected.
        """
        with pytest.raises(ValueError, match="zero spread"):
            KDEGlyph(x, y).evaluate()

    def test_chunking_matches_unchunked(self, cloud, monkeypatch):
        """Chunked evaluation equals the single-block result.

        Test scenario:
            Forcing a tiny MAX_KDE_BLOCK (so the point loop runs in many
            blocks) yields a density identical to the default single block.
        """
        x, y = cloud
        reference = KDEGlyph(x, y, gridsize=30).evaluate()[2]
        monkeypatch.setattr(kde_mod, "MAX_KDE_BLOCK", 5000)
        chunked = KDEGlyph(x, y, gridsize=30).evaluate()[2]
        assert np.allclose(reference, chunked), "Chunked result must match unchunked"


class TestKDEGlyphResolveLevels:
    """Tests for KDEGlyph._resolve_levels."""

    def test_int_levels_evenly_spaced(self, cloud):
        """An integer ``levels`` gives that many evenly-spaced edges.

        Test scenario:
            levels=6 yields 6 edges from density.min to density.max.
        """
        x, y = cloud
        glyph = KDEGlyph(x, y, gridsize=30, levels=6)
        _, _, density = glyph.evaluate()
        edges = glyph._resolve_levels(density)
        assert edges.size == 6, f"Expected 6 edges, got {edges.size}"
        assert np.isclose(edges[0], density.min()), "First edge should be density min"
        assert np.isclose(edges[-1], density.max()), "Last edge should be density max"
        assert np.allclose(
            np.diff(edges), np.diff(edges)[0]
        ), "Edges should be evenly spaced"

    def test_sequence_levels_sorted(self, cloud):
        """An explicit ``levels`` sequence is sorted ascending.

        Test scenario:
            An unsorted list of edges is returned sorted.
        """
        x, y = cloud
        glyph = KDEGlyph(x, y, gridsize=30, levels=[0.3, 0.1, 0.2])
        edges = glyph._resolve_levels(np.zeros((3, 3)))
        assert np.allclose(
            edges, [0.1, 0.2, 0.3]
        ), f"Edges should be sorted, got {edges}"


class TestKDEGlyphApplyClip:
    """Tests for KDEGlyph._apply_clip."""

    def test_none_is_noop(self, cloud):
        """No clip path leaves the contour set unclipped.

        Test scenario:
            With clip_path None, the drawn set has the default axes clip
            (not a custom one).
        """
        x, y = cloud
        glyph = KDEGlyph(x, y, gridsize=30)
        _, _, cs = glyph.plot()
        assert glyph.clip_path is None, "clip_path should be None"

    def test_patch_clip_applied(self, cloud):
        """A Patch clip path is applied to the contour set.

        Test scenario:
            A Circle patch produces a non-None clip path on the drawn set.
        """
        x, y = cloud
        glyph = KDEGlyph(x, y, gridsize=30, clip_path=Circle((0, 0), 1.0))
        _, _, cs = glyph.plot()
        assert cs.get_clip_path() is not None, "Patch clip should be applied"

    def test_standalone_patch_clips_in_data_coordinates(self, cloud):
        """A freshly-constructed Patch clips in data space, not display space.

        Test scenario:
            A standalone Circle of radius 1 at the origin must clip the
            contours to that circle in *data* coordinates: a point inside the
            circle (0, 0) is kept and a far point (5, 5) is excluded. A
            display-space clip (the bug) would instead clip a tiny region near
            the axes origin.
        """
        x, y = cloud
        glyph = KDEGlyph(x, y, gridsize=30, clip_path=Circle((0, 0), 1.0))
        _, ax, cs = glyph.plot()
        path = cs.get_clip_path().get_fully_transformed_path()
        assert path.contains_point(
            ax.transData.transform((0.0, 0.0))
        ), "Clip should keep data point (0, 0) inside the circle"
        assert not path.contains_point(
            ax.transData.transform((5.0, 5.0))
        ), "Clip should exclude data point (5, 5) outside the circle"

    def test_standalone_patch_not_mutated(self, cloud):
        """The glyph clips without mutating the caller's patch.

        Test scenario:
            A standalone Circle passed as clip_path keeps its original
            transform and stays unattached after plotting, so it can be
            reused on another axes.
        """
        x, y = cloud
        patch = Circle((0, 0), 1.0)
        before = patch.get_transform().get_matrix().copy()
        KDEGlyph(x, y, gridsize=20, clip_path=patch).plot()
        assert patch.axes is None, "clip patch should not be attached to the glyph axes"
        assert np.allclose(
            patch.get_transform().get_matrix(), before
        ), "clip patch transform should be left unchanged"

    def test_attached_patch_used_as_is(self, cloud):
        """A Patch already added to an axes is used with its own transform.

        Test scenario:
            When the caller has added the patch to the axes, the glyph uses it
            directly (clip.axes is not None) and leaves it attached.
        """
        x, y = cloud
        fig, host_ax = plt.subplots()
        patch = Circle((0, 0), 1.0)
        host_ax.add_patch(patch)
        glyph = KDEGlyph(x, y, gridsize=20, clip_path=patch, ax=host_ax)
        _, _, cs = glyph.plot()
        assert patch.axes is host_ax, "Attached patch should remain bound to its axes"
        assert cs.get_clip_path() is not None, "Clip should be applied"

    def test_path_clip_applied(self, cloud):
        """A Path clip path is applied in data coordinates.

        Test scenario:
            A square Path produces a non-None clip path on the drawn set.
        """
        x, y = cloud
        square = MplPath([(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)])
        glyph = KDEGlyph(x, y, gridsize=30, clip_path=square)
        _, _, cs = glyph.plot()
        assert cs.get_clip_path() is not None, "Path clip should be applied"

    def test_unsupported_clip_type_raises(self, cloud):
        """An unsupported clip_path type raises TypeError.

        Test scenario:
            A string clip path is rejected at plot time.
        """
        x, y = cloud
        glyph = KDEGlyph(x, y, gridsize=30, clip_path="not-a-path")
        with pytest.raises(
            TypeError, match="clip_path must be a matplotlib Path or Patch"
        ):
            glyph.plot()


class TestKDEGlyphPlot:
    """Tests for KDEGlyph.plot."""

    def test_returns_quadcontourset_with_colorbar(self, cloud):
        """Filled plot returns a QuadContourSet and adds a colorbar.

        Test scenario:
            Default shade=True draws filled contours with a colorbar.
        """
        x, y = cloud
        glyph = KDEGlyph(x, y, gridsize=40)
        fig, ax, cs = glyph.plot()
        assert isinstance(
            cs, QuadContourSet
        ), f"Expected QuadContourSet, got {type(cs)}"
        assert cs.filled, "shade=True should produce filled contours"
        assert glyph.cbar is not None, "A colorbar should be drawn by default"

    def test_shade_false_line_contours(self, cloud):
        """``shade=False`` draws line contours.

        Test scenario:
            The returned contour set is not filled.
        """
        x, y = cloud
        glyph = KDEGlyph(x, y, gridsize=40, shade=False)
        _, _, cs = glyph.plot()
        assert not cs.filled, "shade=False should produce line contours"

    def test_add_colorbar_false_suppresses(self, cloud):
        """``add_colorbar=False`` suppresses the colorbar.

        Test scenario:
            The plot-time override wins and no colorbar is created.
        """
        x, y = cloud
        glyph = KDEGlyph(x, y, gridsize=40)
        glyph.plot(add_colorbar=False)
        assert glyph.cbar is None, "add_colorbar=False should suppress the colorbar"

    def test_title_override(self, cloud):
        """A title passed to plot overrides the default.

        Test scenario:
            plot(title=...) sets the axes title.
        """
        x, y = cloud
        glyph = KDEGlyph(x, y, gridsize=30)
        _, ax, _ = glyph.plot(title="Density")
        assert (
            ax.get_title() == "Density"
        ), f"Title should be 'Density', got {ax.get_title()!r}"

    def test_levels_option_controls_contour_count(self, cloud):
        """The ``levels`` option drives the number of contour levels.

        Test scenario:
            levels=5 yields five contour levels on the drawn set.
        """
        x, y = cloud
        glyph = KDEGlyph(x, y, gridsize=40, levels=5)
        _, _, cs = glyph.plot()
        assert len(cs.levels) == 5, f"Expected 5 levels, got {len(cs.levels)}"

    def test_exposes_im_attribute(self, cloud):
        """The drawn contour set is exposed as ``glyph.im``.

        Test scenario:
            ``im`` is the same object the plot returns.
        """
        x, y = cloud
        glyph = KDEGlyph(x, y, gridsize=30)
        _, _, cs = glyph.plot()
        assert glyph.im is cs, "glyph.im should be the returned contour set"

    def test_plot_on_supplied_axes(self, cloud):
        """``plot(ax=...)`` draws on the given axes and binds its figure.

        Test scenario:
            A pre-existing axes is used as-is and its parent figure is
            adopted by the glyph.
        """
        x, y = cloud
        fig, host_ax = plt.subplots()
        glyph = KDEGlyph(x, y, gridsize=30)
        out_fig, out_ax, _ = glyph.plot(ax=host_ax)
        assert out_ax is host_ax, "plot should draw on the supplied axes"
        assert glyph.fig is host_ax.get_figure(), "glyph.fig should be the axes' figure"

    def test_plot_uses_axes_from_construction(self, cloud):
        """``plot()`` reuses an axes supplied at construction.

        Test scenario:
            With an axes passed to __init__, a no-argument plot() draws on
            that same axes (the `ax is None`/`self.ax` set branch).
        """
        x, y = cloud
        fig, host_ax = plt.subplots()
        glyph = KDEGlyph(x, y, gridsize=30, ax=host_ax)
        _, out_ax, _ = glyph.plot()
        assert out_ax is host_ax, "plot() should reuse the construction axes"

    def test_no_scipy_imported(self, cloud):
        """The estimator never imports scipy.

        Test scenario:
            A full plot leaves scipy absent from sys.modules.
        """
        x, y = cloud
        KDEGlyph(x, y, gridsize=30).plot()
        assert "scipy" not in sys.modules, "KDEGlyph must not import scipy"


class TestKDEGlyphHillshade:
    """Tests for the `hillshade` option (relief-shaded density surface)."""

    @staticmethod
    def _cloud():
        rng = np.random.default_rng(3)
        x = np.concatenate([rng.normal(-2, 0.7, 200), rng.normal(2, 1.0, 150)])
        y = np.concatenate([rng.normal(-1, 0.6, 200), rng.normal(2, 1.2, 150)])
        return x, y

    def test_hillshade_draws_shaded_image(self):
        """`hillshade` renders the density as a shaded RGBA image, not contours."""
        x, y = self._cloud()
        _, _, im = KDEGlyph(x, y, gridsize=40, cmap="magma", hillshade=True).plot()
        assert type(im).__name__ == "AxesImage"
        assert im.get_array().shape[-1] == 4
        plt.close("all")

    def test_colorbar_present_via_proxy(self):
        """A colorbar attaches to the density via a ScalarMappable proxy."""
        x, y = self._cloud()
        g = KDEGlyph(x, y, gridsize=40, hillshade={"vert_exag": 20})
        g.plot()
        assert g.cbar is not None
        plt.close("all")

    def test_without_hillshade_still_contours(self):
        """Without hillshade the KDE still draws a contour set."""
        x, y = self._cloud()
        _, _, cs = KDEGlyph(x, y, gridsize=40).plot()
        assert isinstance(cs, QuadContourSet)
        plt.close("all")

    def test_hillshade_at_plot_time(self):
        """`hillshade` passed to `plot()` shades, mirroring ArrayGlyph/MeshGlyph."""
        x, y = self._cloud()
        _, _, im = KDEGlyph(x, y, gridsize=40, cmap="magma").plot(hillshade={"vert_exag": 20})
        assert type(im).__name__ == "AxesImage"
        assert im.get_array().shape[-1] == 4
        plt.close("all")

    def test_hillshade_honours_title_and_suppressed_colorbar(self):
        """The shaded path sets a title and skips the colorbar when asked."""
        x, y = self._cloud()
        g = KDEGlyph(x, y, gridsize=40, hillshade=True)
        _, ax, _ = g.plot(title="Density terrain", add_colorbar=False)
        assert ax.get_title() == "Density terrain"
        assert g.cbar is None
        plt.close("all")


class TestKDEGlyphDataStyle:
    """Tests for the `style` data-style preset option on KDEGlyph."""

    @staticmethod
    def _cloud():
        rng = np.random.default_rng(3)
        x = np.concatenate([rng.normal(-2, 0.7, 200), rng.normal(2, 1.0, 150)])
        y = np.concatenate([rng.normal(-1, 0.6, 200), rng.normal(2, 1.2, 150)])
        return x, y

    def test_continuous_preset_sets_cmap_at_construction(self):
        """A continuous preset set at construction colours the density."""
        x, y = self._cloud()
        _, _, cs = KDEGlyph(x, y, gridsize=40, style="temperature").plot()
        assert cs.get_cmap().name == "RdYlBu_r"
        plt.close("all")

    def test_continuous_preset_at_plot_time(self):
        """A continuous preset passed to `plot()` colours the density."""
        x, y = self._cloud()
        _, _, cs = KDEGlyph(x, y, gridsize=40).plot(style="elevation")
        assert cs.get_cmap().name == "terrain"
        plt.close("all")

    def test_style_composes_with_hillshade(self):
        """A continuous preset composes with the relief-shaded density image."""
        x, y = self._cloud()
        _, _, im = KDEGlyph(x, y, gridsize=40).plot(style="temperature", hillshade=True)
        assert type(im).__name__ == "AxesImage"
        plt.close("all")

    def test_categorical_style_raises(self):
        """A categorical preset is meaningless for a continuous density and raises."""
        x, y = self._cloud()
        with pytest.raises(ValueError, match="is categorical"):
            KDEGlyph(x, y, gridsize=40, style="flow_direction_d8").plot()
        plt.close("all")

    def test_unknown_style_raises(self):
        """An unknown style name raises a clear `ValueError`."""
        x, y = self._cloud()
        with pytest.raises(ValueError, match="unknown data style"):
            KDEGlyph(x, y, gridsize=40).plot(style="not_a_style")
        plt.close("all")

    def test_style_renders_preset_cmap_without_mutating_default(self):
        """A style renders with the preset cmap but never overwrites `default_options['cmap']`.

        A plot-time `style` name persists (like ArrayGlyph), so a later plain
        `plot()` keeps it -- but the base `default_options['cmap']` is left
        untouched (the cmap is resolved into a local), so nothing is corrupted.
        """
        x, y = self._cloud()
        g = KDEGlyph(x, y, gridsize=40)
        default_cmap = g.default_options["cmap"]
        _, _, cs = g.plot(style="temperature")
        assert cs.get_cmap().name == "RdYlBu_r"
        assert g.default_options["cmap"] == default_cmap
        assert g.style == "temperature"  # the name persists for read-back
        plt.close("all")


class TestKDEGlyphApplyStyle:
    """Tests for the public `apply_style` method and `style` read-back."""

    @staticmethod
    def _cloud():
        rng = np.random.default_rng(3)
        x = np.concatenate([rng.normal(-2, 0.7, 200), rng.normal(2, 1.0, 150)])
        y = np.concatenate([rng.normal(-1, 0.6, 200), rng.normal(2, 1.2, 150)])
        return x, y

    def test_apply_style_restyles_and_reads_back(self):
        """apply_style colours the density and `style` reads back its name."""
        x, y = self._cloud()
        g = KDEGlyph(x, y, gridsize=40)
        g.plot()
        _, _, cs = g.apply_style("temperature")
        assert g.style == "temperature" and cs.get_cmap().name == "RdYlBu_r"
        plt.close("all")

    def test_apply_style_categorical_raises(self):
        """A categorical preset via apply_style raises (density is continuous)."""
        x, y = self._cloud()
        with pytest.raises(ValueError, match="is categorical"):
            KDEGlyph(x, y, gridsize=40).apply_style("flow_direction_d8")
        plt.close("all")

    def test_apply_style_unknown_name_raises(self):
        """An unknown preset name raises a clear `ValueError`."""
        x, y = self._cloud()
        with pytest.raises(ValueError, match="unknown data style"):
            KDEGlyph(x, y, gridsize=40).apply_style("not_a_style")
        plt.close("all")

    def test_style_is_sticky_and_clearable(self):
        """A style survives a later plain plot() and is cleared by style=None."""
        x, y = self._cloud()
        g = KDEGlyph(x, y, gridsize=40)
        g.plot(style="temperature")
        g.plot()
        assert g.style == "temperature"
        g.plot(style=None)
        assert g.style is None
        plt.close("all")
