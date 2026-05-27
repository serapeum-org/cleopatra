"""Tests for cleopatra.vector_glyph.VectorGlyph (T3.1)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.quiver import Barbs, Quiver, QuiverKey

from cleopatra.vector_glyph import VECTOR_DEFAULT_OPTIONS, VectorGlyph


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


@pytest.fixture()
def field():
    """A small 3x3 vector field with a known magnitude range.

    Returns:
        tuple: (x, y, u, v) arrays where u=3, v=4 so |vec| = 5.
    """
    x, y = np.meshgrid(np.arange(3.0), np.arange(3.0))
    u = np.full_like(x, 3.0)
    v = np.full_like(y, 4.0)
    return x, y, u, v


class TestVectorGlyphInit:
    """Tests for VectorGlyph.__init__ and the magnitude property."""

    def test_stores_components_as_arrays(self, field):
        """Inputs are stored as numpy arrays.

        Test scenario:
            All four coordinate/component inputs become ndarrays.
        """
        glyph = VectorGlyph(*field)
        assert all(
            isinstance(a, np.ndarray)
            for a in (glyph.x, glyph.y, glyph.u, glyph.v)
        ), "x/y/u/v should be stored as ndarrays"

    def test_magnitude_is_hypot(self, field):
        """The magnitude property is hypot(u, v).

        Test scenario:
            u=3, v=4 -> every magnitude is 5.
        """
        glyph = VectorGlyph(*field)
        assert np.allclose(glyph.magnitude, 5.0), (
            f"Magnitude should be 5 everywhere, got {glyph.magnitude}"
        )

    def test_ticks_spacing_defaults_none(self, field):
        """ticks_spacing defaults to None for auto-derivation.

        Test scenario:
            The option dict leaves ticks_spacing unset.
        """
        glyph = VectorGlyph(*field)
        assert glyph.default_options["ticks_spacing"] is None, (
            "ticks_spacing should default to None"
        )

    def test_mismatched_uv_raises(self, field):
        """u and v of different shapes raise ValueError.

        Test scenario:
            Shape mismatch between u and v is rejected.
        """
        x, y, u, _ = field
        with pytest.raises(ValueError, match="same shape"):
            VectorGlyph(x, y, u, np.ones(2))

    def test_invalid_kwarg_raises(self, field):
        """An unknown kwarg is rejected by the strict merge.

        Test scenario:
            A key absent from VECTOR_DEFAULT_OPTIONS raises ValueError.
        """
        with pytest.raises(ValueError, match="not correct"):
            VectorGlyph(*field, nope=1)

    def test_cbar_initialized_to_none(self, field):
        """`cbar` is None before plot (L1 fix).

        Test scenario:
            Accessing `cbar` on a fresh glyph returns None rather than
            raising AttributeError, matching ScatterGlyph/PolygonGlyph.
        """
        glyph = VectorGlyph(*field)
        assert glyph.cbar is None, "cbar should be initialized to None in __init__"


class TestVectorGlyphPlot:
    """Tests for VectorGlyph.plot."""

    def test_quiver_returns_quiver_with_magnitude(self, field):
        """quiver returns a Quiver carrying the magnitude array.

        Test scenario:
            kind='quiver' -> Quiver mappable whose array is the
            magnitude (5 everywhere) and a colorbar is attached.
        """
        glyph = VectorGlyph(*field)
        fig, ax, im = glyph.plot(kind="quiver")
        assert isinstance(im, Quiver), f"Expected Quiver, got {type(im)}"
        assert np.allclose(im.get_array(), 5.0), "Quiver array should be the magnitude"
        assert glyph.cbar is not None, "A colorbar should be attached"

    def test_barbs_returns_barbs(self, field):
        """barbs returns a Barbs mappable.

        Test scenario:
            kind='barbs' -> Barbs artist with the magnitude array.
        """
        glyph = VectorGlyph(*field)
        _, _, im = glyph.plot(kind="barbs")
        assert isinstance(im, Barbs), f"Expected Barbs, got {type(im)}"
        assert np.allclose(im.get_array(), 5.0), "Barbs array should be the magnitude"

    def test_streamplot_returns_linecollection_with_array(self):
        """streamplot returns a LineCollection carrying the magnitude.

        Test scenario:
            On a regular grid, kind='streamplot' yields lines whose
            colour array is populated (non-None).
        """
        y, x = np.mgrid[0:5, 0:5].astype(float)
        u = np.ones_like(x)
        v = np.ones_like(y)
        glyph = VectorGlyph(x, y, u, v)
        _, _, im = glyph.plot(kind="streamplot")
        assert im.get_array() is not None, "Streamplot lines should carry a colour array"
        assert glyph.cbar is not None, "A colorbar should be attached"

    def test_streamplot_clim_pinned_to_tick_range(self):
        """streamplot colour limits match the tick range (L2 fix).

        Test scenario:
            On the linear path, the LineCollection clim equals the
            colorbar tick range (ticks[0], ticks[-1]) so colours and the
            colorbar agree — matching the quiver/barbs behaviour.
        """
        y, x = np.mgrid[0:5, 0:5].astype(float)
        u = np.full_like(x, 3.0)
        v = np.full_like(y, 4.0)
        glyph = VectorGlyph(x, y, u, v, vmin=0.0, vmax=10.0, ticks_spacing=2.0)
        _, _, im = glyph.plot(kind="streamplot")
        ticks = glyph.get_ticks()
        assert im.get_clim() == (ticks[0], ticks[-1]), (
            f"streamplot clim should equal the tick range, got {im.get_clim()}"
        )

    def test_quiver_clim_pinned_to_tick_range(self):
        """quiver colour limits also match the tick range (parity check).

        Test scenario:
            quiver's clim equals (ticks[0], ticks[-1]) on the linear
            path, the behaviour streamplot is aligned to.
        """
        x, y = np.meshgrid(np.arange(3.0), np.arange(3.0))
        u = np.full_like(x, 3.0)
        v = np.full_like(y, 4.0)
        glyph = VectorGlyph(x, y, u, v, vmin=0.0, vmax=10.0, ticks_spacing=2.0)
        _, _, im = glyph.plot(kind="quiver")
        ticks = glyph.get_ticks()
        assert im.get_clim() == (ticks[0], ticks[-1]), (
            f"quiver clim should equal the tick range, got {im.get_clim()}"
        )

    def test_unknown_kind_raises(self, field):
        """An unrecognised kind raises ValueError before drawing.

        Test scenario:
            kind='blah' is rejected with a helpful message.
        """
        glyph = VectorGlyph(*field)
        with pytest.raises(ValueError, match="unknown vector kind"):
            glyph.plot(kind="blah")

    def test_levels_produce_boundary_norm(self, field):
        """An integer `levels` discretises the magnitude colour scale.

        Test scenario:
            levels=3 -> the quiver norm is a BoundaryNorm.
        """
        glyph = VectorGlyph(*field, levels=3, vmin=0.0, vmax=6.0)
        _, _, im = glyph.plot(kind="quiver")
        assert isinstance(im.norm, mcolors.BoundaryNorm), (
            f"levels should yield a BoundaryNorm, got {type(im.norm)}"
        )

    def test_plot_on_supplied_axes(self, field):
        """Plotting onto a supplied axes reuses that axes/figure.

        Test scenario:
            Passing ax to plot draws on it and returns its figure.
        """
        fig, ax = plt.subplots()
        glyph = VectorGlyph(*field)
        out_fig, out_ax, _ = glyph.plot(ax=ax)
        assert out_ax is ax, "Should draw on the supplied axes"
        assert out_fig is fig, "Should return the supplied axes' figure"

    def test_title_override(self, field):
        """A title passed to plot is applied to the axes.

        Test scenario:
            plot(title=...) sets the axes title.
        """
        glyph = VectorGlyph(*field)
        _, ax, _ = glyph.plot(title="Wind")
        assert ax.get_title() == "Wind", f"Unexpected title: {ax.get_title()}"

    def test_scale_forwarded_to_quiver(self, field):
        """The `scale` option is forwarded to quiver.

        Test scenario:
            scale=50 -> the Quiver's scale attribute is 50.
        """
        glyph = VectorGlyph(*field, scale=50)
        _, _, im = glyph.plot(kind="quiver")
        assert im.scale == 50, f"Expected quiver scale 50, got {im.scale}"


class TestVectorGlyphAddKey:
    """Tests for VectorGlyph.add_key."""

    def test_add_key_returns_quiverkey(self, field):
        """add_key returns a QuiverKey with the requested label.

        Test scenario:
            A labelled key is created for a quiver plot.
        """
        glyph = VectorGlyph(*field)
        _, _, im = glyph.plot(kind="quiver")
        key = glyph.add_key(im, value=5.0, label="5 m/s")
        assert isinstance(key, QuiverKey), f"Expected QuiverKey, got {type(key)}"
        assert key.text.get_text() == "5 m/s", f"Unexpected label: {key.text.get_text()}"

    def test_add_key_default_label_is_value(self, field):
        """With no label, the numeric value is rendered.

        Test scenario:
            add_key(value=7) -> label text '7'.
        """
        glyph = VectorGlyph(*field)
        _, _, im = glyph.plot(kind="quiver")
        key = glyph.add_key(im, value=7.0)
        assert key.text.get_text() == "7", f"Default label should be '7', got {key.text.get_text()}"


def test_vector_default_options_extend_style_defaults():
    """VECTOR_DEFAULT_OPTIONS is a superset of the shared style defaults.

    Test scenario:
        The module-level options dict carries both the base style keys
        and the vector-specific additions.
    """
    assert "figsize" in VECTOR_DEFAULT_OPTIONS, "Should inherit base style keys"
    assert "density" in VECTOR_DEFAULT_OPTIONS, "Should add vector keys"


class TestAddColorbarToggle:
    """`add_colorbar=False` suppresses VectorGlyph's colorbar (#3)."""

    @staticmethod
    def _field():
        gx, gy = np.meshgrid(np.arange(5), np.arange(5))
        return gx, gy, np.ones_like(gx, float), np.ones_like(gx, float)

    def test_default_draws_colorbar(self):
        """A vector field draws its magnitude colorbar by default."""
        gx, gy, u, v = self._field()
        glyph = VectorGlyph(gx, gy, u, v)
        fig, ax, _ = glyph.plot()
        try:
            assert glyph.cbar is not None, "default should draw a colorbar"
            assert len(fig.axes) == 2, f"expected 2 axes, got {len(fig.axes)}"
        finally:
            plt.close(fig)

    def test_add_colorbar_false_suppresses(self):
        """`add_colorbar=False` leaves cbar None and adds no axes."""
        gx, gy, u, v = self._field()
        glyph = VectorGlyph(gx, gy, u, v, add_colorbar=False)
        fig, ax, _ = glyph.plot()
        try:
            assert glyph.cbar is None, "add_colorbar=False should skip the colorbar"
            assert len(fig.axes) == 1, f"expected 1 axes, got {len(fig.axes)}"
        finally:
            plt.close(fig)

    def test_plot_time_override_suppresses(self):
        """Passing `add_colorbar=False` to `plot` suppresses the colorbar.

        Test scenario:
            Plot-time override: even with the default construction option,
            `plot(add_colorbar=False)` draws no colorbar.
        """
        gx, gy, u, v = self._field()
        glyph = VectorGlyph(gx, gy, u, v)
        fig, ax, _ = glyph.plot(add_colorbar=False)
        try:
            assert glyph.cbar is None, "plot(add_colorbar=False) should skip the colorbar"
            assert len(fig.axes) == 1, f"expected 1 axes, got {len(fig.axes)}"
        finally:
            plt.close(fig)

    def test_add_colorbar_in_option_keys(self):
        """`add_colorbar` is an accepted option key."""
        assert "add_colorbar" in VectorGlyph.option_keys(), "add_colorbar missing"
