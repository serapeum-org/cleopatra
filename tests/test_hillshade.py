"""Tests for `cleopatra.hillshade` (relief-shading primitives).

Covers the option resolver and the two geometry-specific shaders -- `shade_grid`
(regular raster) and `shade_faces` (triangulated mesh) -- with deterministic
numpy inputs (no plotting).
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.colors import Normalize

from cleopatra.hillshade import (
    DEFAULT_HILLSHADE,
    resolve_hillshade,
    shade_faces,
    shade_grid,
)


class TestResolveHillshade:
    """Tests for `resolve_hillshade`."""

    def test_falsy_is_off(self):
        """`False`/`None`/`0` mean the feature is off."""
        assert resolve_hillshade(False) is None
        assert resolve_hillshade(None) is None

    def test_true_uses_all_defaults(self):
        """`True` returns a full copy of the defaults."""
        opts = resolve_hillshade(True)
        assert opts == DEFAULT_HILLSHADE
        assert opts is not DEFAULT_HILLSHADE  # a copy, not the shared dict

    def test_dict_overrides_a_subset(self):
        """A dict overrides only the given keys, keeping the rest at default."""
        opts = resolve_hillshade({"vert_exag": 5, "azimuth": 300})
        assert opts["vert_exag"] == 5 and opts["azimuth"] == 300
        assert opts["altitude"] == DEFAULT_HILLSHADE["altitude"]

    def test_unknown_key_raises(self):
        """An unknown option key raises `ValueError`."""
        with pytest.raises(ValueError, match="unknown hillshade options"):
            resolve_hillshade({"bogus": 1})

    def test_bad_blend_mode_raises(self):
        """An invalid `blend_mode` raises `ValueError`."""
        with pytest.raises(ValueError, match="blend_mode"):
            resolve_hillshade({"blend_mode": "nope"})

    @pytest.mark.parametrize("value", [False, 4, [0, 90, 180]])
    def test_multidirectional_valid_forms(self, value):
        """False, an int, and a sequence are all accepted for `multidirectional`."""
        assert resolve_hillshade({"multidirectional": value})["multidirectional"] == value

    @pytest.mark.parametrize("bad", [True, "north", 1.5])
    def test_multidirectional_invalid_forms_raise_clearly(self, bad):
        """A bool, string, or non-iterable float `multidirectional` raises `ValueError`."""
        with pytest.raises(ValueError, match="multidirectional"):
            resolve_hillshade({"multidirectional": bad})


class TestShadeGrid:
    """Tests for `shade_grid` (raster relief shading)."""

    @pytest.fixture
    def dem(self):
        """A small varied synthetic DEM."""
        yy, xx = np.mgrid[0:20, 0:25]
        return 10.0 + 0.5 * yy + 8.0 * np.sin(xx / 3.0) * np.cos(yy / 3.0)

    def test_returns_rgba_image(self, dem):
        """The result is an `(H, W, 4)` RGBA image."""
        rgba = shade_grid(dem, "terrain")
        assert rgba.shape == dem.shape + (4,)

    def test_nan_is_transparent(self, dem):
        """A NaN elevation cell is fully transparent; finite cells opaque."""
        dem = dem.copy()
        dem[0, 0] = np.nan
        rgba = shade_grid(dem, "terrain")
        assert rgba[0, 0, 3] == 0.0
        assert rgba[10, 12, 3] == 1.0

    def test_shading_changes_the_image(self, dem):
        """Relief shading changes the pixels vs. a plain colormapping."""
        plain = mpl.colormaps["terrain"](Normalize()(dem))
        shaded = shade_grid(dem, "terrain", vert_exag=5)
        assert not np.allclose(shaded[..., :3], plain[..., :3])

    def test_multidirectional_differs_from_single(self, dem):
        """A multidirectional hillshade differs from the single-azimuth one."""
        single = shade_grid(dem, "terrain", vert_exag=5)
        multi = shade_grid(dem, "terrain", vert_exag=5, multidirectional=4)
        assert not np.allclose(single, multi)

    def test_multidirectional_sequence_of_azimuths(self, dem):
        """An explicit sequence of azimuths averages those directions."""
        rgba = shade_grid(dem, "terrain", vert_exag=5, multidirectional=[0.0, 90.0, 180.0])
        assert rgba.shape == dem.shape + (4,)
        assert np.all(np.isfinite(rgba))


class TestShadeFaces:
    """Tests for `shade_faces` (mesh relief shading)."""

    @pytest.fixture
    def quad(self):
        """A unit quad split into two triangles, with flat face colours."""
        verts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        facecolors = np.tile([0.4, 0.6, 0.3, 1.0], (2, 1))
        return verts, faces, facecolors

    def test_returns_rgba_per_face(self, quad):
        """The result is one RGBA per face, with alpha preserved."""
        verts, faces, facecolors = quad
        z = np.array([0.0, 0.0, 1.0, 1.0])
        shaded = shade_faces(verts, faces, z, facecolors)
        assert shaded.shape == (2, 4)
        np.testing.assert_allclose(shaded[:, 3], 1.0)

    def test_flat_surface_shades_uniformly(self, quad):
        """A flat surface (equal z) illuminates every face identically."""
        verts, faces, facecolors = quad
        z = np.full(4, 3.0)  # flat
        shaded = shade_faces(verts, faces, z, facecolors)
        np.testing.assert_allclose(shaded[0], shaded[1])

    def test_tilted_surface_changes_colour(self, quad):
        """A tilted surface shades the faces away from their flat colour."""
        verts, faces, facecolors = quad
        z = np.array([0.0, 0.0, 2.0, 2.0])  # slope in +y
        shaded = shade_faces(verts, faces, z, facecolors, vert_exag=1)
        assert not np.allclose(shaded[:, :3], facecolors[:, :3])

    def test_ignores_grid_only_kwargs(self, quad):
        """Grid-only kwargs (dx/dy/multidirectional) are accepted and ignored."""
        verts, faces, facecolors = quad
        z = np.array([0.0, 0.0, 1.0, 1.0])
        shaded = shade_faces(
            verts, faces, z, facecolors, dx=2.0, dy=2.0, multidirectional=3, fraction=1.0
        )
        assert shaded.shape == (2, 4)
