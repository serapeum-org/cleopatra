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
        with pytest.raises(ValueError, match=r"1-D lon/lat"):
            ArrayGlyph(data, projection="globe").plot()
        plt.close("all")

    def test_two_dimensional_coords_projection_raises(self, field):
        """2-D curvilinear coords are rejected for a projection preset.

        Test scenario:
            ``apply_projection_style`` needs 1-D vectors, so 2-D coordinate
            meshes raise the same precondition error.
        """
        data, lon, lat = field
        lon2d, lat2d = np.meshgrid(lon, lat)
        with pytest.raises(ValueError, match=r"1-D lon/lat"):
            ArrayGlyph(data, coords=(lon2d, lat2d), projection="globe").plot()
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
