"""Tests for the ``projection`` preset kwarg on ``ArrayGlyph``.

Covers the ``"flat"`` and ``"globe"`` projection presets, the 1-D-coords
precondition, and that omitting ``projection`` is unchanged. The globe path
needs ``pyproj`` (the ``[tiles]`` extra) and is guarded with
``pytest.importorskip``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.image import AxesImage

from cleopatra.array_glyph import ArrayGlyph

pytestmark = pytest.mark.plot


@pytest.fixture(scope="function")
def field():
    """Provide a small lon/lat field with 1-D coordinate vectors.

    Returns:
        tuple: ``(data, lon, lat)`` where ``data`` is ``(6, 8)`` and ``lon`` /
        ``lat`` are 1-D centre vectors of length 8 and 6.
    """
    lon = np.linspace(-10.0, 10.0, 8)
    lat = np.linspace(30.0, 50.0, 6)
    data = np.random.default_rng(0).random((6, 8))
    return data, lon, lat


class TestArrayGlyphProjection:
    """Tests for ``ArrayGlyph.plot(projection=...)``."""

    def test_flat_projection_renders_quadmesh(self, field):
        """``projection='flat'`` draws a ``QuadMesh`` at the projected edges.

        Test scenario:
            The flat preset needs no pyproj and produces a pcolormesh mappable.
        """
        data, lon, lat = field
        glyph = ArrayGlyph(data, coords=(lon, lat), projection="flat")
        glyph.plot()
        assert isinstance(glyph.im, QuadMesh), f"expected a QuadMesh, got {type(glyph.im).__name__}"
        plt.close("all")

    def test_globe_projection_renders_and_frames(self, field):
        """``projection='globe'`` reprojects and draws the globe frame.

        Test scenario:
            With pyproj present, the globe preset renders a ``QuadMesh`` and adds
            the boundary patch to the axes.
        """
        pytest.importorskip("pyproj", reason="globe projection needs the [tiles] extra")
        data, lon, lat = field
        glyph = ArrayGlyph(data, coords=(lon, lat), projection="globe")
        glyph.plot()
        assert isinstance(glyph.im, QuadMesh), f"expected a QuadMesh, got {type(glyph.im).__name__}"
        assert len(glyph.ax.patches) >= 1, "globe render should add a boundary patch"
        plt.close("all")

    def test_extent_only_projection_raises(self, field):
        """A projection without coords raises a clear ``ValueError``.

        Test scenario:
            An extent-only glyph has no lon/lat to reproject, so
            ``projection='globe'`` is rejected before any drawing.
        """
        data, _lon, _lat = field
        glyph = ArrayGlyph(data, projection="globe")
        with pytest.raises(ValueError, match=r"1-D lon/lat"):
            glyph.plot()
        plt.close("all")

    def test_two_dimensional_coords_projection_raises(self, field):
        """2-D curvilinear coords are rejected for a projection preset.

        Test scenario:
            ``apply_projection_style`` needs 1-D vectors, so 2-D coordinate
            meshes raise the same precondition error.
        """
        data, lon, lat = field
        lon2d, lat2d = np.meshgrid(lon, lat)
        glyph = ArrayGlyph(data, coords=(lon2d, lat2d), projection="globe")
        with pytest.raises(ValueError, match=r"1-D lon/lat"):
            glyph.plot()
        plt.close("all")

    def test_no_projection_is_unchanged(self, field):
        """Omitting ``projection`` keeps the default raster (``AxesImage``).

        Test scenario:
            The default ``projection=None`` renders via imshow as before.
        """
        data, _lon, _lat = field
        glyph = ArrayGlyph(data)
        glyph.plot()
        assert isinstance(glyph.im, AxesImage), f"expected an AxesImage, got {type(glyph.im).__name__}"
        plt.close("all")

    @pytest.mark.parametrize("revert", [None, "flat"])
    def test_globe_then_flat_restores_view(self, field, revert):
        """A plain replot after a globe un-freezes the reused axes (MF2).

        Args:
            revert: The projection cleared to on the second plot.

        Test scenario:
            `ArrayGlyph` reuses its axes; a globe freezes the view to ±R metres
            and hides the axis, so a later flat replot must restore the degree-
            scale view + axis and drop the globe boundary, or the flat map is an
            invisible speck.
        """
        pytest.importorskip("pyproj", reason="globe projection needs the [tiles] extra")
        data, lon, lat = field
        glyph = ArrayGlyph(data, coords=(lon, lat))
        glyph.plot(projection="globe")
        assert glyph.ax.get_xlim()[1] > 1e6, "globe should freeze the view to metres"
        glyph.plot(projection=revert)
        assert glyph.ax.get_xlim()[1] <= 360.0, "flat replot should restore the degree-scale view"
        assert glyph.ax.axison, "flat replot should turn the axis back on"
        assert len(glyph.ax.patches) == 0, "flat replot should remove the globe boundary"
        plt.close("all")

    @pytest.mark.parametrize("revert", [None, "flat"])
    def test_styled_globe_then_flat_restores_view(self, field, revert):
        """A styled replot after a styled globe un-freezes the axes (MF2, A4).

        Args:
            revert: The projection cleared to on the second plot.

        Test scenario:
            The styled-globe (A4) path shares the same frozen-view failure mode;
            a subsequent flat styled render must restore the view + axis.
        """
        pytest.importorskip("pyproj", reason="globe projection needs the [tiles] extra")
        data, lon, lat = field
        scaled = data * 30.0
        glyph = ArrayGlyph(scaled, coords=(lon, lat))
        glyph.plot(style="temperature_2m", projection="globe")
        assert glyph.ax.get_xlim()[1] > 1e6, "styled globe should freeze the view"
        glyph.plot(style="temperature_2m", projection=revert)
        assert glyph.ax.get_xlim()[1] <= 360.0, "styled flat replot should restore the view"
        assert glyph.ax.axison, "styled flat replot should turn the axis back on"
        assert len(glyph.ax.patches) == 0, "styled flat replot should remove the globe boundary"
        plt.close("all")

    def test_globe_replot_does_not_stack_frame(self, field):
        """Re-plotting a globe on the same glyph does not stack boundary patches.

        Test scenario:
            The prior frame is stripped before the new one is drawn, so the patch
            count stays constant across two globe renders.
        """
        pytest.importorskip("pyproj", reason="globe projection needs the [tiles] extra")
        data, lon, lat = field
        glyph = ArrayGlyph(data, coords=(lon, lat))
        glyph.plot(projection="globe")
        patches1 = len(glyph.ax.patches)
        glyph.plot(projection="globe")
        assert len(glyph.ax.patches) == patches1, "globe boundary must not stack on replot"
        plt.close("all")
