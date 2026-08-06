"""Tests for the ``projection`` preset kwarg on ``MeshGlyph``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cleopatra.glyphs.mesh_glyph import MeshGlyph

pytestmark = pytest.mark.plot


@pytest.fixture(scope="function")
def mesh():
    """Provide a small lon/lat triangular mesh and per-node values.

    Returns:
        tuple: ``(node_x, node_y, faces, values)`` covering a European lon/lat box.
    """
    lons = np.linspace(-20.0, 40.0, 9)
    lats = np.linspace(30.0, 65.0, 7)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    node_x = grid_lon.ravel()
    node_y = grid_lat.ravel()
    n = len(lons)
    faces = []
    for j in range(len(lats) - 1):
        for i in range(len(lons) - 1):
            a = j * n + i
            faces.append([a, a + 1, a + n + 1])
            faces.append([a, a + n + 1, a + n])
    return node_x, node_y, np.array(faces), node_x + node_y


class TestMeshGlyphProjection:
    """Tests for ``MeshGlyph(projection=...)``."""

    def test_globe_reprojects_and_frames(self, mesh):
        """``projection="globe"`` reprojects the nodes and draws the frame.

        Test scenario:
            The axes ends up in projected metres (~orthographic radius) and a
            boundary patch is added.
        """
        pytest.importorskip("pyproj", reason="globe needs the [tiles] extra")
        node_x, node_y, faces, values = mesh
        glyph = MeshGlyph(node_x, node_y, faces, projection="globe")
        glyph.plot(values, location="node")
        assert len(glyph.ax.patches) >= 1, "globe render should add a boundary patch"
        assert glyph.ax.get_xlim()[1] > 1e6, "axes should be in projected metres"
        plt.close("all")

    def test_projection_via_plot_kwarg(self, mesh):
        """`projection` may be passed to `plot()` as well as the constructor.

        Test scenario:
            `plot(..., projection="globe")` frames the axes.
        """
        pytest.importorskip("pyproj", reason="globe needs the [tiles] extra")
        node_x, node_y, faces, values = mesh
        glyph = MeshGlyph(node_x, node_y, faces)
        glyph.plot(values, location="node", projection="globe")
        assert len(glyph.ax.patches) >= 1, "plot(projection='globe') should frame the axes"
        plt.close("all")

    def test_sticky_projection_survives_replot(self, mesh):
        """A constructor `projection` survives a later plain `plot(data)`.

        Test scenario:
            Like the sticky `style`, `projection` persists across a second
            `plot()` that does not pass it.
        """
        pytest.importorskip("pyproj", reason="globe needs the [tiles] extra")
        node_x, node_y, faces, values = mesh
        glyph = MeshGlyph(node_x, node_y, faces, projection="globe")
        glyph.plot(values, location="node")
        glyph.plot(values, location="node")  # no projection= here
        assert len(glyph.ax.patches) >= 1, "sticky projection should persist across replot"
        plt.close("all")

    def test_replot_does_not_stack_globe_frame(self, mesh):
        """A sticky-globe replot reuses the frame instead of stacking a duplicate.

        Test scenario:
            The boundary patch and graticule lines are removed and redrawn each
            render, so the exact patch/line counts are identical across two
            plots (they used to double: 1->2 patches, 15->30 lines).
        """
        pytest.importorskip("pyproj", reason="globe needs the [tiles] extra")
        node_x, node_y, faces, values = mesh
        glyph = MeshGlyph(node_x, node_y, faces, projection="globe")
        glyph.plot(values, location="node")
        patches1, lines1 = len(glyph.ax.patches), len(glyph.ax.lines)
        glyph.plot(values, location="node")
        assert len(glyph.ax.patches) == patches1, "globe boundary must not stack on replot"
        assert len(glyph.ax.lines) == lines1, "graticule must not stack on replot"
        plt.close("all")

    @pytest.mark.parametrize("revert", [None, "flat"])
    def test_clear_projection_restores_flat_coords_and_view(self, mesh, revert):
        """Reverting a globe to `None`/`"flat"` restores flat coords AND the view.

        Args:
            revert: The projection value that clears the globe (`None` or
                `"flat"`, both flat views).

        Test scenario:
            Reverting drops the reprojected triangulation cache, removes the
            globe frame, and — critically — un-freezes the axes: the view returns
            to lon/lat degrees (not the ±R orthographic metres the globe froze)
            and the axis is turned back on, so the flat mesh is actually visible.
        """
        pytest.importorskip("pyproj", reason="globe needs the [tiles] extra")
        node_x, node_y, faces, values = mesh
        glyph = MeshGlyph(node_x, node_y, faces, projection="globe")
        glyph.plot(values, location="node")
        assert glyph.triangulation.x.max() > 1e6, "globe should reproject to metres"
        assert glyph.ax.get_xlim()[1] > 1e6, "globe should freeze the view to metres"
        glyph.plot(values, location="node", projection=revert)
        assert glyph.triangulation.x.max() <= 360.0, "revert should restore lon/lat coords"
        assert glyph.ax.get_xlim()[1] <= 360.0, "revert should restore the flat view"
        assert glyph.ax.axison, "revert should turn the axis back on"
        assert len(glyph.ax.patches) == 0, "revert should remove the globe frame"
        plt.close("all")

    @pytest.mark.parametrize("projection", ["flat", None])
    def test_no_globe_frame_for_flat_or_none(self, mesh, projection):
        """`"flat"` and no projection draw no globe boundary.

        Args:
            projection: `"flat"` or `None`.

        Test scenario:
            Only `"globe"` adds a boundary patch.
        """
        node_x, node_y, faces, values = mesh
        glyph = MeshGlyph(node_x, node_y, faces, projection=projection)
        glyph.plot(values, location="node")
        assert len(glyph.ax.patches) == 0, f"projection={projection!r} should draw no boundary"
        plt.close("all")
