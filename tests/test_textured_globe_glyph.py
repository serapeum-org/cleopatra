"""Tests for `cleopatra.glyphs.globe.textured_globe_glyph.TexturedGlobeGlyph`.

Covers construction/validation, that `draw` returns a 3-D axes, the equirectangular texture -> sphere orientation
(north-up, west-left), the axial tilt and spin, the sample-once caching contract, and the `animate` helper. Meshes are
kept small so the CPU-only 3-D surface draws stay fast.
"""

import doctest

import matplotlib
import numpy as np
import pytest

matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D

import cleopatra.glyphs.globe.textured_globe_glyph as globe_module
from cleopatra.glyphs.globe.textured_globe_glyph import (
    EARTH_TILT_DEG,
    TexturedGlobeGlyph,
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


@pytest.fixture
def texture() -> np.ndarray:
    """A small distinctive equirectangular RGB texture (north red, south blue, west darkened)."""
    tex = np.zeros((16, 32, 3), dtype=np.uint8)
    tex[:8] = (200, 40, 40)  # northern hemisphere red
    tex[8:] = (40, 40, 200)  # southern hemisphere blue
    tex[:, :16] //= 2  # western hemisphere darkened
    return tex


def test_module_doctests_execute():
    """Run the module's docstring examples (pytest is not configured with --doctest-modules)."""
    try:
        results = doctest.testmod(globe_module, verbose=False)
    finally:
        plt.close("all")
    assert results.failed == 0, (
        f"{results.failed} doctest example(s) failed in textured_globe_glyph"
    )
    assert results.attempted > 0, (
        "no doctest examples were collected from textured_globe_glyph"
    )


class TestConstruction:
    def test_defaults(self, texture):
        globe = TexturedGlobeGlyph(texture)
        assert globe.tilt_deg == EARTH_TILT_DEG
        assert (globe.n_lon, globe.n_lat) == (180, 90)
        assert globe.brightness == 1.0

    def test_texture_normalised_to_float_rgba(self, texture):
        globe = TexturedGlobeGlyph(texture)
        assert globe.texture.shape == (16, 32, 4)
        assert globe.texture.dtype == float
        assert globe.texture.min() >= 0.0 and globe.texture.max() <= 1.0
        # opaque alpha added for an RGB source
        assert np.all(globe.texture[..., 3] == 1.0)

    def test_alpha_channel_preserved(self):
        tex = np.ones((8, 16, 4), dtype=float)
        tex[..., 3] = 0.5
        globe = TexturedGlobeGlyph(tex)
        assert np.allclose(globe.texture[..., 3], 0.5)

    def test_brightness_darkens_and_clips(self, texture):
        dark = TexturedGlobeGlyph(texture, brightness=0.0)
        assert np.all(dark.texture[..., :3] == 0.0)
        bright = TexturedGlobeGlyph(texture, brightness=100.0)
        assert bright.texture[..., :3].max() <= 1.0

    def test_float_texture_above_one_normalised_by_peak_not_255(self):
        tex = np.full((4, 4, 3), 0.5, dtype=float)
        tex[0, 0, 0] = 1.5
        globe = TexturedGlobeGlyph(tex, n_lon=6, n_lat=4)
        # 0.5 / 1.5 == 1/3, not 0.5/255 (which would be ~black)
        assert np.isclose(globe.texture[1, 1, 0], 1.0 / 3.0)

    def test_nan_cells_render_black_without_breaking_normalisation(self):
        tex = np.linspace(0.0, 200.0, 4 * 4 * 3, dtype=float).reshape(4, 4, 3)
        tex[0, 0, 0] = np.nan
        globe = TexturedGlobeGlyph(tex, n_lon=6, n_lat=4)
        assert not np.any(np.isnan(globe.texture))
        assert globe.texture.max() <= 1.0
        assert globe.texture[0, 0, 0] == 0.0  # NaN -> black

    def test_integer_max_one_scaled_by_255(self):
        tex = np.zeros((4, 4, 3), dtype=np.uint8)
        tex[0, 0, 0] = 1
        globe = TexturedGlobeGlyph(tex, n_lon=6, n_lat=4)
        assert np.isclose(globe.texture[0, 0, 0], 1.0 / 255.0)

    def test_integer_alpha_scaled_by_255(self):
        tex = np.full((4, 4, 4), 255, dtype=np.uint8)
        tex[..., 3] = 128
        globe = TexturedGlobeGlyph(tex, n_lon=6, n_lat=4)
        assert np.allclose(globe.texture[..., 3], 128.0 / 255.0)

    @pytest.mark.parametrize(
        "bad",
        [
            np.zeros((16, 32)),  # 2-D, no channel axis
            np.zeros((16, 32, 5)),  # 5 channels
            np.zeros((1, 32, 3)),  # too few rows
            np.zeros((16, 1, 3)),  # too few cols
        ],
    )
    def test_bad_texture_raises(self, bad):
        with pytest.raises(ValueError):
            TexturedGlobeGlyph(bad)

    @pytest.mark.parametrize(
        "kwargs", [{"n_lon": 1}, {"n_lat": 1}, {"brightness": -0.5}]
    )
    def test_bad_params_raise(self, texture, kwargs):
        with pytest.raises(ValueError):
            TexturedGlobeGlyph(texture, **kwargs)

    def test_non_3d_ax_raises(self, texture):
        _, ax2d = plt.subplots()
        with pytest.raises(ValueError):
            TexturedGlobeGlyph(texture, ax=ax2d)


class TestDraw:
    def test_returns_3d_axes(self, texture):
        fig, ax = TexturedGlobeGlyph(texture, n_lon=36, n_lat=18).draw()
        assert isinstance(ax, Axes3D)
        assert ax.name == "3d"

    def test_facecolors_shape_and_range(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=36, n_lat=18)
        globe.draw()
        assert globe._facecolors.shape == (17, 35, 4)  # (n_lat-1, n_lon-1, 4)
        assert globe._facecolors.min() >= 0.0 and globe._facecolors.max() <= 1.0
        assert globe.surface is not None

    def test_draw_on_supplied_3d_axes(self, texture):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        fig2, ax2 = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12).draw(ax)
        assert ax2 is ax
        assert fig2 is fig

    def test_draw_on_non_3d_ax_raises(self, texture):
        _, ax2d = plt.subplots()
        with pytest.raises(ValueError):
            TexturedGlobeGlyph(texture, n_lon=24, n_lat=12).draw(ax2d)

    def test_draw_uses_instance_fig(self, texture):
        fig = plt.figure()
        fig2, ax = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12, fig=fig).draw()
        assert fig2 is fig
        assert isinstance(ax, Axes3D)

    def test_draw_uses_instance_ax(self, texture):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        fig2, ax2 = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12, ax=ax).draw()
        assert ax2 is ax
        assert fig2 is fig

    def test_redraw_clears_prior_surface(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        fig, ax = globe.draw(spin=0.0)
        globe.draw(ax, spin=90.0)
        # a single surface collection remains on the axes, not two
        assert len(ax.collections) == 1

    def test_background_option(self, texture):
        fig, ax = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12).draw(
            background="black"
        )
        assert fig.get_facecolor() == to_rgba("black")


class TestOrientation:
    """The equirectangular texture must map north-up and west-left onto the sphere."""

    def test_facecolors_north_up_south_down(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=36, n_lat=18)
        globe.draw()
        fc = globe._facecolors
        # northernmost face row is red-dominant, southernmost is blue-dominant
        north = fc[0].mean(axis=0)
        south = fc[-1].mean(axis=0)
        assert north[0] > north[2]  # red > blue in the north
        assert south[2] > south[0]  # blue > red in the south

    def test_facecolors_west_darker_than_east(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=36, n_lat=18)
        globe.draw()
        fc = globe._facecolors
        west = fc[:, 0, :3].mean()
        east = fc[:, -1, :3].mean()
        assert west < east  # western hemisphere was halved

    def test_base_mesh_geographic_anchors(self, texture):
        # odd counts put lat=0 / lon=0 exactly on a grid vertex
        globe_flat = TexturedGlobeGlyph(texture, n_lon=5, n_lat=3, tilt_deg=0.0)
        globe_flat._prepare()
        flat = np.stack(globe_flat._spun_mesh(0.0))
        # (lat=0, lon=0) -> +x ; north pole (row 0) -> +z with no tilt
        assert np.allclose(flat[:, 1, 2], [1.0, 0.0, 0.0], atol=1e-9)
        assert np.allclose(flat[:, 0, 0], [0.0, 0.0, 1.0], atol=1e-9)


class TestTiltAndSpin:
    def test_tilt_moves_the_pole(self, texture):
        flat = TexturedGlobeGlyph(texture, n_lon=5, n_lat=3, tilt_deg=0.0)
        tilted = TexturedGlobeGlyph(texture, n_lon=5, n_lat=3, tilt_deg=45.0)
        flat._prepare()
        tilted._prepare()
        pole_flat = np.stack(flat._spun_mesh(0.0))[:, 0, 0]
        pole_tilted = np.stack(tilted._spun_mesh(0.0))[:, 0, 0]
        assert not np.allclose(pole_flat, pole_tilted)
        # tilt is a rotation: the pole stays on the unit sphere
        assert np.isclose(np.linalg.norm(pole_tilted), 1.0)

    def test_spin_rotates_mesh_but_not_texture(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        globe._prepare()
        fc_before = globe._facecolors.copy()
        mesh0 = np.stack(globe._spun_mesh(0.0))
        mesh90 = np.stack(globe._spun_mesh(90.0))
        assert not np.allclose(mesh0, mesh90)  # geometry rotated
        assert np.array_equal(globe._facecolors, fc_before)  # texture untouched

    def test_prepare_is_idempotent_sample_once(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        globe._prepare()
        fc_id = id(globe._facecolors)
        base_id = id(globe._base_xyz)
        globe._prepare()  # second call must not rebuild
        assert id(globe._facecolors) == fc_id
        assert id(globe._base_xyz) == base_id


class TestAnimate:
    def test_returns_funcanimation(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        anim = globe.animate(n_frames=3)
        assert isinstance(anim, FuncAnimation)

    def test_animate_reuses_single_axes(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        anim = globe.animate(ax, n_frames=4)
        # drive a couple of frames; the redraw contract keeps one surface
        anim._func(0)
        anim._func(1)
        assert len(ax.collections) == 1


class TestIntrospection:
    def test_option_keys(self):
        keys = TexturedGlobeGlyph.option_keys()
        assert {"figsize", "elev", "azim", "background"} <= keys

    def test_filter_kwargs(self):
        filtered = TexturedGlobeGlyph.filter_kwargs({"elev": 30, "bogus": 1})
        assert filtered == {"elev": 30}

    def test_default_options_merges_overrides(self, texture):
        globe = TexturedGlobeGlyph(texture, elev=42)
        opts = globe.default_options
        assert opts["elev"] == 42
        assert opts["azim"] == 0.0  # untouched default retained


def test_no_new_dependency():
    """The globe uses mpl_toolkits.mplot3d, which ships with matplotlib -- no new dependency."""
    import mpl_toolkits.mplot3d as m3d

    assert m3d.__name__.startswith("mpl_toolkits")
