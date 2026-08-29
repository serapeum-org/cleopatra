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


@pytest.mark.filterwarnings("ignore:Animation was deleted without rendering anything")
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
        assert globe.texture.min() >= 0.0
        assert globe.texture.max() <= 1.0
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

    def test_uint16_texture_scaled_by_dtype_max(self):
        tex = np.zeros((4, 4, 3), dtype=np.uint16)
        tex[0, 0, 0] = 30000
        globe = TexturedGlobeGlyph(tex, n_lon=6, n_lat=4)
        assert np.isclose(globe.texture[0, 0, 0], 30000.0 / 65535.0)

    def test_integer_alpha_scaled_by_255(self):
        tex = np.full((4, 4, 4), 255, dtype=np.uint8)
        tex[..., 3] = 128
        globe = TexturedGlobeGlyph(tex, n_lon=6, n_lat=4)
        assert np.allclose(globe.texture[..., 3], 128.0 / 255.0)

    def test_float_rgba_peak_above_one_preserves_alpha(self):
        tex = np.full((4, 4, 4), 0.5, dtype=float)
        tex[0, 0, 0] = 2.0  # an RGB value > 1 triggers peak normalization
        globe = TexturedGlobeGlyph(tex, n_lon=6, n_lat=4)
        # only RGB is scaled by the peak; alpha stays as supplied
        assert np.allclose(globe.texture[..., 3], 0.5)

    def test_all_nan_texture_renders_black_without_warning(self, recwarn):
        tex = np.full((4, 4, 3), np.nan, dtype=float)
        globe = TexturedGlobeGlyph(tex, n_lon=6, n_lat=4)
        assert not any(issubclass(w.category, RuntimeWarning) for w in recwarn)
        assert np.all(globe.texture[..., :3] == 0.0)

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

    def test_unknown_kwarg_raises(self, texture):
        with pytest.raises(ValueError, match="Unknown option"):
            TexturedGlobeGlyph(texture, elevv=80)


class TestDraw:
    def test_returns_3d_axes(self, texture):
        fig, ax = TexturedGlobeGlyph(texture, n_lon=36, n_lat=18).draw()
        assert isinstance(ax, Axes3D)
        assert ax.name == "3d"

    def test_facecolors_shape_and_range(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=36, n_lat=18)
        globe.draw()
        assert globe._facecolors.shape == (17, 35, 4)  # (n_lat-1, n_lon-1, 4)
        assert globe._facecolors.min() >= 0.0
        assert globe._facecolors.max() <= 1.0
        assert globe.surface is not None

    def test_draw_on_supplied_3d_axes(self, texture):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        fig2, ax2 = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12).draw(ax)
        assert ax2 is ax
        assert fig2 is fig

    def test_draw_on_non_3d_ax_raises(self, texture):
        _, ax2d = plt.subplots()
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        with pytest.raises(ValueError):
            globe.draw(ax2d)

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

    def test_draw_unknown_kwarg_raises(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        with pytest.raises(ValueError, match="Unknown option"):
            globe.draw(elevv=80)


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
    @pytest.mark.filterwarnings(
        "ignore:Animation was deleted without rendering anything"
    )
    def test_returns_funcanimation(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        anim = globe.animate(n_frames=3)
        assert isinstance(anim, FuncAnimation)

    def test_animate_unknown_kwarg_raises(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        with pytest.raises(ValueError, match="Unknown option"):
            globe.animate(n_frames=3, elevv=80)

    def test_animate_reuses_single_axes(self, texture, tmp_path):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        anim = globe.animate(ax, n_frames=4)
        assert list(anim.new_frame_seq()) == [0, 1, 2, 3]
        # render every frame via the public writer; each frame's update is draw(),
        # whose redraw keeps a single surface on the reused axes
        anim.save(tmp_path / "globe.gif", writer="pillow", fps=5)
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


class TestLighting:
    def test_sun_defaults_to_none_even(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        assert globe.sun is None
        globe._prepare()
        # sun=None returns the cache unchanged -- byte-identical to a 0.33.0 unlit render
        lit = globe._lit_facecolors(globe._spun_mesh(0.0), None, globe.ambient)
        assert lit is globe._facecolors

    def test_default_ambient(self, texture):
        assert TexturedGlobeGlyph(texture).ambient == pytest.approx(0.13)

    def test_sun_normalised_to_unit(self, texture):
        globe = TexturedGlobeGlyph(texture, sun=(3.0, 0.0, 0.0))
        assert np.allclose(globe.sun, [1.0, 0.0, 0.0])

    def test_lighting_produces_terminator(self):
        tex = np.full((8, 16, 3), 200, np.uint8)
        globe = TexturedGlobeGlyph(tex, n_lon=48, n_lat=24)
        globe._prepare()
        lit = globe._lit_facecolors(
            globe._spun_mesh(0.0), np.array([1.0, 0.0, 0.0]), 0.13
        )
        rgb = lit[..., :3]
        assert (
            rgb.min() < 0.5 * rgb.max()
        )  # a real day/night range on a uniform texture

    def test_ambient_floor_not_black(self):
        tex = np.full((8, 16, 3), 200, np.uint8)
        globe = TexturedGlobeGlyph(tex, n_lon=24, n_lat=12)
        globe._prepare()
        cache_max = float(globe._facecolors[..., :3].max())
        lit = globe._lit_facecolors(
            globe._spun_mesh(0.0), np.array([1.0, 0.0, 0.0]), 0.2
        )
        assert float(lit[..., :3].min()) == pytest.approx(0.2 * cache_max, abs=1e-6)

    def test_lit_fraction_tracks_spin(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12, sun=(1.0, 0.0, 0.0))
        globe._prepare()
        lit0 = globe._lit_facecolors(globe._spun_mesh(0.0), globe.sun, globe.ambient)
        lit180 = globe._lit_facecolors(
            globe._spun_mesh(180.0), globe.sun, globe.ambient
        )
        assert not np.allclose(
            lit0, lit180
        )  # fixed sun, spinning globe -> terminator moves

    def test_cache_not_mutated_by_lighting(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12, sun=(1.0, 0.0, 0.0))
        globe._prepare()
        before = globe._facecolors.copy()
        globe.draw()
        assert np.array_equal(
            globe._facecolors, before
        )  # no re-sample / mutation per frame

    def test_draw_sun_none_disables_instance_light(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12, sun=(1.0, 0.0, 0.0))
        globe._prepare()
        lit = globe._lit_facecolors(globe._spun_mesh(0.0), None, globe.ambient)
        assert lit is globe._facecolors

    def test_draw_accepts_sun_on_init_and_per_call(self, texture):
        fig, ax = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12, sun=(1, 0, 0)).draw()
        assert ax.name == "3d"
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        globe.draw(sun=(0, 0, 1), ambient=0.25)
        assert globe.surface is not None

    def test_animate_forwards_sun(self, texture, tmp_path):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        anim = globe.animate(ax, n_frames=3, sun=(1.0, 0.0, 0.0))
        anim.save(tmp_path / "lit.gif", writer="pillow", fps=5)
        assert len(ax.collections) == 1

    @pytest.mark.parametrize(
        "bad_sun", [(1.0, 0.0), (0.0, 0.0, 0.0), (np.nan, 0.0, 0.0), (1, 2, 3, 4)]
    )
    def test_bad_sun_raises(self, texture, bad_sun):
        with pytest.raises(ValueError):
            TexturedGlobeGlyph(texture, sun=bad_sun)

    @pytest.mark.parametrize("bad_ambient", [-0.1, 1.5])
    def test_bad_ambient_raises(self, texture, bad_ambient):
        with pytest.raises(ValueError):
            TexturedGlobeGlyph(texture, ambient=bad_ambient)

    def test_draw_bad_sun_raises(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        with pytest.raises(ValueError):
            globe.draw(sun=(0.0, 0.0, 0.0))

    @pytest.mark.parametrize("bad_sun", [[[1.0, 0.0, 0.0]], np.zeros((3, 1)), 1.0])
    def test_non_1d_sun_raises(self, texture, bad_sun):
        with pytest.raises(ValueError):
            TexturedGlobeGlyph(texture, sun=bad_sun)

    def test_ambient_nan_raises(self, texture):
        with pytest.raises(ValueError):
            TexturedGlobeGlyph(texture, ambient=float("nan"))

    def test_ambient_zero_gives_black_night(self):
        tex = np.full((8, 16, 3), 200, np.uint8)
        globe = TexturedGlobeGlyph(tex, n_lon=24, n_lat=12)
        globe._prepare()
        lit = globe._lit_facecolors(
            globe._spun_mesh(0.0), np.array([1.0, 0.0, 0.0]), 0.0
        )
        assert float(lit[..., :3].min()) == pytest.approx(
            0.0, abs=1e-9
        )  # night -> black

    def test_ambient_one_equals_cache(self):
        tex = np.full((8, 16, 3), 200, np.uint8)
        globe = TexturedGlobeGlyph(tex, n_lon=24, n_lat=12)
        globe._prepare()
        lit = globe._lit_facecolors(
            globe._spun_mesh(0.0), np.array([1.0, 0.0, 0.0]), 1.0
        )
        assert np.allclose(lit, globe._facecolors)  # no dimming at ambient=1

    def test_animate_validates_sun_eagerly(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        with pytest.raises(
            ValueError
        ):  # raised at the animate() call, not at frame render
            globe.animate(ax, n_frames=3, sun=(0.0, 0.0, 0.0))

    def test_animate_validates_ambient_eagerly(self, texture):
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        with pytest.raises(ValueError):
            globe.animate(ax, n_frames=3, ambient=5.0)

    def test_world_space_sun_honoured_under_tilt(self):
        # sun is world-space (+z), not body-space. With a 45deg tilt the geographic north
        # cap leans toward +z (so north > south), but it is NOT the brightest region --
        # an equatorial face pointing straight at world +z is. A body-space regression
        # (dotting the un-tilted normals) would instead light the pole fully (north == peak).
        tex = np.full((16, 32, 3), 200, np.uint8)
        globe = TexturedGlobeGlyph(tex, n_lon=48, n_lat=24, tilt_deg=45.0)
        globe._prepare()
        lit = globe._lit_facecolors(
            globe._spun_mesh(0.0), np.array([0.0, 0.0, 1.0]), 0.1
        )
        north = float(lit[0, :, :3].mean())
        south = float(lit[-1, :, :3].mean())
        peak = float(lit[..., :3].max())
        assert north > south  # tilt leans the north cap toward the light
        assert (
            north < peak - 0.1
        )  # but the pole is not the peak -> world-space, not body-space


class TestTiltTransform:
    def test_rotation_matrix_identity_without_tilt_or_spin(self, texture):
        globe = TexturedGlobeGlyph(texture, tilt_deg=0.0)
        assert np.allclose(globe.rotation_matrix(0.0), np.eye(3))

    def test_rotation_matrix_is_tilt_then_spin(self, texture):
        globe = TexturedGlobeGlyph(texture, tilt_deg=30.0)
        expected = TexturedGlobeGlyph._rotation_x(
            30.0
        ) @ TexturedGlobeGlyph._rotation_z(47.0)
        assert np.allclose(globe.rotation_matrix(47.0), expected)

    def test_transform_lands_where_the_mesh_does(self, texture):
        # DoD: a point pushed through the exposed transform lands where the glyph's own mesh puts it
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12, tilt_deg=30.0)
        globe._prepare()
        spin = 47.0
        mesh = np.stack(globe._spun_mesh(spin)).reshape(3, -1).T  # (N, 3) world points
        out = globe.transform(
            globe._base_xyz.T, spin=spin
        )  # base points through the public transform
        assert np.allclose(out, mesh)

    def test_transform_single_point_shape_and_value(self, texture):
        globe = TexturedGlobeGlyph(texture, tilt_deg=90.0)
        out = globe.transform([0.0, 0.0, 1.0])  # north pole under a 90deg x-tilt -> -y
        assert out.shape == (3,)
        assert np.allclose(out, [0.0, -1.0, 0.0])

    def test_transform_array_applies_per_row(self, texture):
        globe = TexturedGlobeGlyph(texture, tilt_deg=30.0)
        pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        out = globe.transform(pts, spin=12.0)
        assert out.shape == (3, 3)
        # each row is transformed like a single point (not merely reshaped)
        for row_in, row_out in zip(pts, out):
            assert np.allclose(row_out, globe.transform(row_in, spin=12.0))

    @pytest.mark.parametrize(
        "bad",
        [np.zeros(2), np.zeros((4, 2)), np.zeros((2, 3, 3)), 5.0, np.array(5.0)],
    )
    def test_transform_bad_shape_raises(self, texture, bad):
        # scalar / 0-d must raise ValueError (not IndexError from indexing shape[-1])
        globe = TexturedGlobeGlyph(texture)
        with pytest.raises(ValueError):
            globe.transform(bad)

    def test_default_tilt_mesh_unchanged(self, texture):
        # the refactor keeps the X-axis default: the mesh equals R_x(tilt) @ R_z(spin) @ base
        globe = TexturedGlobeGlyph(texture, n_lon=24, n_lat=12)
        globe._prepare()
        expected = TexturedGlobeGlyph._rotation_x(globe.tilt_deg) @ (
            TexturedGlobeGlyph._rotation_z(15.0) @ globe._base_xyz
        )
        actual = np.stack(globe._spun_mesh(15.0)).reshape(3, -1)
        assert np.allclose(actual, expected)

    def test_transform_empty_array_preserved(self, texture):
        out = TexturedGlobeGlyph(texture).transform(np.zeros((0, 3)))
        assert out.shape == (0, 3)

    def test_transform_1x3_not_squeezed(self, texture):
        out = TexturedGlobeGlyph(texture).transform([[1.0, 0.0, 0.0]])
        assert out.shape == (1, 3)

    def test_transform_accepts_list_input(self, texture):
        globe = TexturedGlobeGlyph(texture, tilt_deg=20.0)
        from_list = globe.transform([0.0, 0.0, 1.0], spin=30.0)
        from_array = globe.transform(np.array([0.0, 0.0, 1.0]), spin=30.0)
        assert np.allclose(from_list, from_array)

    def test_rotation_matrix_orthogonal_and_fresh(self, texture):
        globe = TexturedGlobeGlyph(texture, tilt_deg=23.44)
        m = globe.rotation_matrix(47.0)
        assert np.allclose(m @ m.T, np.eye(3))  # orthogonal
        assert np.isclose(np.linalg.det(m), 1.0)  # a proper rotation
        m[0, 0] = 9.0  # mutating the returned matrix must not corrupt a later call
        assert not np.allclose(globe.rotation_matrix(47.0), m)

    def test_transform_works_before_prepare(self, texture):
        globe = TexturedGlobeGlyph(texture)
        assert globe._base_xyz is None  # never drawn / prepared
        assert globe.transform([0.0, 0.0, 1.0], spin=10.0).shape == (3,)

    def test_transform_output_independent_of_input(self, texture):
        globe = TexturedGlobeGlyph(texture, tilt_deg=0.0)
        inp = np.array([1.0, 2.0, 3.0])
        out = globe.transform(inp, spin=0.0)
        out[0] = 99.0
        assert inp[0] == 1.0  # the returned array does not alias the input


def test_no_new_dependency():
    """The globe uses mpl_toolkits.mplot3d, which ships with matplotlib -- no new dependency."""
    import mpl_toolkits.mplot3d as m3d

    assert m3d.__name__.startswith("mpl_toolkits")
