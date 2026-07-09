"""Tests for cleopatra.scatter_glyph.ScatterGlyph (T2.1)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection

from cleopatra.scatter_glyph import SCATTER_DEFAULT_OPTIONS, ScatterGlyph


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


@pytest.fixture()
def xy():
    """A small, fixed set of point coordinates.

    Returns:
        tuple[np.ndarray, np.ndarray]: x and y coordinate arrays.
    """
    return np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.0, 1.0])


@pytest.fixture()
def values():
    """A per-point value array aligned with the `xy` fixture.

    Returns:
        np.ndarray: Four scalar values spanning 0..30.
    """
    return np.array([0.0, 10.0, 20.0, 30.0])


class TestScatterGlyphInit:
    """Tests for ScatterGlyph.__init__."""

    def test_stores_coordinates_as_arrays(self, xy):
        """Coordinates are stored as numpy arrays.

        Test scenario:
            Plain lists are accepted and converted via np.asarray.
        """
        glyph = ScatterGlyph([0, 1, 2], [3, 4, 5])
        assert isinstance(glyph.x, np.ndarray), "x should be an ndarray"
        assert isinstance(glyph.y, np.ndarray), "y should be an ndarray"
        assert glyph.values is None, "values should default to None"

    def test_options_include_scatter_keys(self, xy):
        """Scatter-specific option keys are present with their defaults.

        Test scenario:
            marker/point_size defaults are exposed and ticks_spacing is
            None so the shared helper auto-derives it.
        """
        glyph = ScatterGlyph(*xy)
        assert glyph.default_options["marker"] == "o", "Default marker should be 'o'"
        assert glyph.default_options["point_size"] == 20, "Default size should be 20"
        assert (
            glyph.default_options["ticks_spacing"] is None
        ), "ticks_spacing should default to None for auto-derivation"

    def test_kwargs_override_options(self, xy):
        """Constructor kwargs override option defaults.

        Test scenario:
            marker and point_size passed as kwargs win over the defaults.
        """
        glyph = ScatterGlyph(*xy, marker="^", point_size=50)
        assert glyph.default_options["marker"] == "^", "marker kwarg should win"
        assert glyph.default_options["point_size"] == 50, "point_size kwarg should win"

    def test_invalid_kwarg_raises(self, xy):
        """An unknown kwarg is rejected by the strict merge.

        Test scenario:
            A key absent from SCATTER_DEFAULT_OPTIONS raises ValueError.
        """
        with pytest.raises(ValueError, match="not correct"):
            ScatterGlyph(*xy, not_a_real_option=1)

    def test_mismatched_xy_raises(self):
        """x and y of different lengths raise ValueError.

        Test scenario:
            len(x) != len(y) is rejected at construction.
        """
        with pytest.raises(ValueError, match="same shape"):
            ScatterGlyph([0, 1, 2], [0, 1])

    def test_mismatched_values_raises(self, xy):
        """A values array not matching x/y length raises ValueError.

        Test scenario:
            len(values) != len(x) is rejected at construction.
        """
        with pytest.raises(ValueError, match="values must match"):
            ScatterGlyph(*xy, values=np.array([1.0, 2.0]))


class TestScatterGlyphPlot:
    """Tests for ScatterGlyph.plot."""

    def test_uncoloured_returns_pathcollection_no_array(self, xy):
        """Uncoloured points return a PathCollection without a value array.

        Test scenario:
            With no values, scatter has no colour array and no colorbar.
        """
        glyph = ScatterGlyph(*xy)
        fig, ax, paths = glyph.plot()
        assert isinstance(
            paths, PathCollection
        ), f"Expected PathCollection, got {type(paths)}"
        assert paths.get_array() is None, "Uncoloured scatter should carry no array"
        assert glyph.cbar is None, "No colorbar should be created without values"

    def test_coloured_maps_values_and_adds_colorbar(self, xy, values):
        """Coloured points carry the value array and gain a colorbar.

        Test scenario:
            The scatter's array equals the input values and a colorbar
            is attached.
        """
        glyph = ScatterGlyph(*xy, values=values)
        fig, ax, paths = glyph.plot()
        np.testing.assert_array_almost_equal(
            paths.get_array(), values, err_msg="Scatter array should equal values"
        )
        assert (
            glyph.cbar is not None
        ), "A colorbar should be created for coloured points"

    def test_point_size_and_marker_forwarded(self, xy):
        """point_size maps to the scatter sizes.

        Test scenario:
            point_size=80 -> every path size equals 80.
        """
        glyph = ScatterGlyph(*xy, point_size=80)
        _, _, paths = glyph.plot()
        sizes = paths.get_sizes()
        assert np.all(sizes == 80), f"Expected all sizes 80, got {sizes}"

    def test_auto_limits_set_from_values(self, xy, values):
        """vmin/vmax are auto-resolved from the value range.

        Test scenario:
            Unset limits resolve to (min, max) of the values.
        """
        glyph = ScatterGlyph(*xy, values=values)
        glyph.plot()
        assert glyph.vmin == 0.0, f"vmin should be data min 0.0, got {glyph.vmin}"
        assert glyph.vmax == 30.0, f"vmax should be data max 30.0, got {glyph.vmax}"

    def test_explicit_limits_preserved(self, xy, values):
        """Explicit vmin/vmax (with pinned spacing) drive the clim.

        Test scenario:
            vmin/vmax pinned via kwargs, plus a ticks_spacing that
            divides the range evenly, yield a clim equal to those
            limits. The clim follows ticks[0]/ticks[-1] (the documented
            contract shared with ArrayGlyph); an evenly-dividing spacing
            keeps the top tick at exactly vmax.
        """
        glyph = ScatterGlyph(
            *xy, values=values, vmin=0.0, vmax=40.0, ticks_spacing=10.0
        )
        _, _, paths = glyph.plot()
        assert paths.get_clim() == (
            0.0,
            40.0,
        ), f"clim should honour explicit limits, got {paths.get_clim()}"

    def test_levels_produce_boundary_norm(self, xy, values):
        """An integer `levels` discretises the colour scale.

        Test scenario:
            levels=4 -> the scatter norm is a BoundaryNorm.
        """
        glyph = ScatterGlyph(*xy, values=values, levels=4)
        _, _, paths = glyph.plot()
        assert isinstance(
            paths.norm, mcolors.BoundaryNorm
        ), f"levels should yield a BoundaryNorm, got {type(paths.norm)}"

    def test_power_color_scale_applied(self, xy, values):
        """A non-linear color_scale is honoured for points.

        Test scenario:
            color_scale='power' -> the scatter norm is a PowerNorm.
        """
        glyph = ScatterGlyph(*xy, values=values, color_scale="power")
        _, _, paths = glyph.plot()
        assert isinstance(
            paths.norm, mcolors.PowerNorm
        ), f"color_scale='power' should yield a PowerNorm, got {type(paths.norm)}"

    def test_plot_on_supplied_axes(self, xy, values):
        """Plotting onto a supplied axes reuses that axes/figure.

        Test scenario:
            Passing ax to plot draws on it and returns its figure.
        """
        fig, ax = plt.subplots()
        glyph = ScatterGlyph(*xy, values=values)
        out_fig, out_ax, _ = glyph.plot(ax=ax)
        assert out_ax is ax, "Should draw on the supplied axes"
        assert out_fig is fig, "Should return the supplied axes' figure"

    def test_title_override(self, xy):
        """A title passed to plot is applied to the axes.

        Test scenario:
            plot(title=...) sets the axes title.
        """
        glyph = ScatterGlyph(*xy)
        _, ax, _ = glyph.plot(title="My Points")
        assert ax.get_title() == "My Points", f"Unexpected title: {ax.get_title()}"

    def test_all_nan_values_raise(self, xy):
        """All-NaN values with unpinned limits raise ValueError.

        Test scenario:
            The shared helper rejects an unusable colour range.
        """
        glyph = ScatterGlyph(*xy, values=np.full(4, np.nan))
        with pytest.raises(ValueError, match="no finite values"):
            glyph.plot()


def test_scatter_default_options_extend_style_defaults():
    """SCATTER_DEFAULT_OPTIONS is a superset of the shared style defaults.

    Test scenario:
        The module-level options dict carries both the base style keys
        and the scatter-specific additions.
    """
    assert "figsize" in SCATTER_DEFAULT_OPTIONS, "Should inherit base style keys"
    assert "point_size" in SCATTER_DEFAULT_OPTIONS, "Should add scatter keys"


class TestAddColorbarToggle:
    """`add_colorbar=False` suppresses ScatterGlyph's colorbar (#3)."""

    @staticmethod
    def _xyv():
        return (
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
        )

    def test_default_draws_colorbar(self):
        """By default a coloured scatter draws its colorbar (extra axes)."""
        x, y, v = self._xyv()
        glyph = ScatterGlyph(x, y, values=v)
        fig, ax, _ = glyph.plot()
        try:
            assert glyph.cbar is not None, "default should draw a colorbar"
            assert len(fig.axes) == 2, f"expected 2 axes, got {len(fig.axes)}"
        finally:
            plt.close(fig)

    def test_add_colorbar_false_suppresses(self):
        """`add_colorbar=False` leaves cbar None and adds no axes."""
        x, y, v = self._xyv()
        glyph = ScatterGlyph(x, y, values=v, add_colorbar=False)
        fig, ax, _ = glyph.plot()
        try:
            assert glyph.cbar is None, "add_colorbar=False should skip the colorbar"
            assert len(fig.axes) == 1, f"expected 1 axes, got {len(fig.axes)}"
        finally:
            plt.close(fig)

    def test_plot_time_override_suppresses(self):
        """Passing `add_colorbar=False` to `plot` suppresses the colorbar.

        Test scenario:
            Plot-time override mirrors ArrayGlyph: even with the default
            construction option, `plot(add_colorbar=False)` draws no colorbar.
        """
        x, y, v = self._xyv()
        glyph = ScatterGlyph(x, y, values=v)
        fig, ax, _ = glyph.plot(add_colorbar=False)
        try:
            assert (
                glyph.cbar is None
            ), "plot(add_colorbar=False) should skip the colorbar"
            assert len(fig.axes) == 1, f"expected 1 axes, got {len(fig.axes)}"
        finally:
            plt.close(fig)

    def test_plot_time_override_enables(self):
        """`plot(add_colorbar=True)` draws even when constructed with False.

        Test scenario:
            The plot-time override works in both directions — a glyph built
            with `add_colorbar=False` can still draw a colorbar per-call.
        """
        x, y, v = self._xyv()
        glyph = ScatterGlyph(x, y, values=v, add_colorbar=False)
        fig, ax, _ = glyph.plot(add_colorbar=True)
        try:
            assert glyph.cbar is not None, "plot(add_colorbar=True) should draw"
            assert len(fig.axes) == 2, f"expected 2 axes, got {len(fig.axes)}"
        finally:
            plt.close(fig)

    def test_plot_time_none_keeps_construction_value(self):
        """`add_colorbar=None` (default) keeps the construction-time setting.

        Test scenario:
            Omitting the override leaves the constructor's `add_colorbar=False`
            in force.
        """
        x, y, v = self._xyv()
        glyph = ScatterGlyph(x, y, values=v, add_colorbar=False)
        fig, ax, _ = glyph.plot()
        try:
            assert glyph.cbar is None, "construction-time False should persist"
        finally:
            plt.close(fig)

    def test_plot_time_override_does_not_persist(self):
        """A plot-time `add_colorbar` override does not mutate the glyph options.

        Test scenario:
            `plot(add_colorbar=False)` suppresses the colorbar for that call
            only; the glyph's `default_options["add_colorbar"]` stays True so a
            later `plot()` draws again.
        """
        x, y, v = self._xyv()
        glyph = ScatterGlyph(x, y, values=v)
        fig, ax, _ = glyph.plot(add_colorbar=False)
        plt.close(fig)
        assert (
            glyph.default_options["add_colorbar"] is True
        ), "override must not persist into default_options"
        fig, ax, _ = glyph.plot()
        try:
            assert glyph.cbar is not None, "a later plot() should draw again"
        finally:
            plt.close(fig)

    def test_add_colorbar_in_option_keys(self):
        """`add_colorbar` is an accepted option key (no validation error)."""
        assert "add_colorbar" in ScatterGlyph.option_keys(), "add_colorbar missing"
