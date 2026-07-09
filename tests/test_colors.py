from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from matplotlib.image import AxesImage

from cleopatra.colors import (
    CAMS_COLORMAPS,
    DATA_STYLES,
    Colors,
    alpha_scaled_image,
    alpha_scaled_mesh,
    apply_data_style,
)


class TestCamsColormaps:
    """Tests for the `CAMS_COLORMAPS` preset constant."""

    def test_has_organic_matter_and_dust(self):
        """The two documented preset names are present, and only those two."""
        assert set(CAMS_COLORMAPS) == {"organic_matter", "dust"}, (
            f"unexpected preset names: {set(CAMS_COLORMAPS)}"
        )

    @pytest.mark.parametrize("name", ["organic_matter", "dust"])
    def test_entries_are_colormaps(self, name):
        """Each entry is a ready `Colormap`, not a name string or dict."""
        assert isinstance(CAMS_COLORMAPS[name], Colormap), (
            f"{name} is not a Colormap: {type(CAMS_COLORMAPS[name])}"
        )

    @pytest.mark.parametrize("name", ["organic_matter", "dust"])
    def test_starts_white_at_zero(self, name):
        """Every CAMS colormap starts at opaque white for value 0.0."""
        assert CAMS_COLORMAPS[name](0.0) == (1.0, 1.0, 1.0, 1.0), (
            f"{name}(0.0) should be white, got {CAMS_COLORMAPS[name](0.0)}"
        )

    def test_dust_ends_dark_brown(self):
        """The dust colormap saturates to a dark brown at value 1.0."""
        r, g, b, a = CAMS_COLORMAPS["dust"](1.0)
        assert a == 1.0, "alpha should be opaque"
        assert r > g > b, f"dust top stop should be brown-toned, got rgb=({r}, {g}, {b})"

    def test_organic_matter_ends_purple(self):
        """The organic_matter colormap saturates to a deep purple at value 1.0."""
        r, g, b, a = CAMS_COLORMAPS["organic_matter"](1.0)
        assert a == 1.0, "alpha should be opaque"
        assert r > g and b > g, (
            f"organic_matter top stop should be purple-toned, got rgb=({r}, {g}, {b})"
        )


class TestAlphaScaledImage:
    """Tests for `alpha_scaled_image`."""

    @pytest.fixture
    def ax(self):
        """A fresh Axes on the Agg backend, closed after the test."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_returns_axes_image(self, ax):
        """The call returns an `AxesImage` attached to the given axes."""
        img = alpha_scaled_image(ax, np.array([[0.0, 1.0]]), "viridis")
        assert isinstance(img, AxesImage), f"expected AxesImage, got {type(img)}"
        assert img in ax.images, "image should be attached to the given axes"

    def test_alpha_matches_normalised_value(self, ax):
        """Alpha equals the (default-normalised) data value, not the colour."""
        data = np.array([[0.0, 0.25, 1.0]])
        img = alpha_scaled_image(ax, data, "viridis")
        alpha = img.get_array()[..., 3]
        np.testing.assert_allclose(
            alpha, [[0.0, 0.25, 1.0]], err_msg=f"unexpected alpha channel: {alpha}"
        )

    def test_nan_is_always_fully_transparent(self, ax):
        """A NaN cell is alpha=0 even under an alpha_norm that would not zero it."""
        data = np.array([[np.nan, 5.0]])
        img = alpha_scaled_image(
            ax, data, "viridis", alpha_norm=Normalize(vmin=0.0, vmax=5.0, clip=False)
        )
        alpha = img.get_array()[..., 3]
        assert alpha[0, 0] == 0.0, f"NaN pixel should be transparent, got alpha={alpha[0, 0]}"
        assert alpha[0, 1] == 1.0, f"finite max value should be opaque, got {alpha[0, 1]}"

    def test_decoupled_alpha_norm(self, ax):
        """A separate `alpha_norm` drives opacity independently of `norm`."""
        data = np.array([[0.0, 10.0]])
        img = alpha_scaled_image(
            ax,
            data,
            "viridis",
            norm=Normalize(vmin=0.0, vmax=10.0),
            alpha_norm=Normalize(vmin=0.0, vmax=20.0),
        )
        alpha = img.get_array()[..., 3]
        np.testing.assert_allclose(
            alpha, [[0.0, 0.5]], err_msg=f"alpha_norm override not applied: {alpha}"
        )

    def test_cmap_accepts_colormap_object(self, ax):
        """A `Colormap` instance (not just a name string) is accepted directly."""
        img = alpha_scaled_image(ax, np.array([[0.0, 1.0]]), CAMS_COLORMAPS["dust"])
        rgb = img.get_array()[0, 1, :3]
        np.testing.assert_allclose(
            rgb, [0.36, 0.17, 0.02], atol=0.01, err_msg=f"unexpected top-stop colour: {rgb}"
        )

    def test_non_2d_data_raises(self, ax):
        """A 1D (or higher-dimensional) `data` array raises `ValueError`."""
        with pytest.raises(ValueError, match="2-dimensional"):
            alpha_scaled_image(ax, np.array([0.0, 1.0]), "viridis")

    def test_forwards_imshow_kwargs(self, ax):
        """Extra keyword arguments (e.g. `zorder`) reach the underlying `imshow`."""
        img = alpha_scaled_image(ax, np.array([[0.0, 1.0]]), "viridis", zorder=7)
        assert img.get_zorder() == 7, f"zorder not forwarded, got {img.get_zorder()}"


class TestAlphaScaledMesh:
    """Tests for `alpha_scaled_mesh` (the curvilinear-grid counterpart)."""

    @pytest.fixture
    def ax(self):
        """A fresh Axes on the Agg backend, closed after the test."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    @pytest.fixture
    def xy(self):
        """A 3x3 corner grid for a 2x2 quad mesh (shading='flat' convention)."""
        return np.meshgrid(np.arange(3), np.arange(3))

    def test_returns_quadmesh_attached_to_ax(self, ax, xy):
        """The call returns a `QuadMesh` registered as a collection on `ax`."""
        x, y = xy
        data = np.array([[0.0, 1.0], [0.5, 0.25]])
        mesh = alpha_scaled_mesh(ax, x, y, data, "viridis", shading="flat")
        assert isinstance(mesh, QuadMesh), f"expected QuadMesh, got {type(mesh)}"
        assert mesh in ax.collections, "mesh should be attached to ax"

    def test_facecolor_alpha_matches_normalised_value(self, ax, xy):
        """Per-quad alpha in `facecolor` equals the (default-normalised) value."""
        x, y = xy
        data = np.array([[0.0, 1.0], [0.5, 0.25]])
        mesh = alpha_scaled_mesh(ax, x, y, data, "viridis", shading="flat")
        alpha = mesh.get_facecolor()[:, 3]
        np.testing.assert_allclose(
            alpha, [0.0, 1.0, 0.5, 0.25], err_msg=f"unexpected facecolor alpha: {alpha}"
        )

    def test_nan_cell_is_fully_transparent(self, ax, xy):
        """A NaN cell renders with alpha=0 in the facecolor array."""
        x, y = xy
        data = np.array([[np.nan, 1.0], [0.5, 0.25]])
        mesh = alpha_scaled_mesh(ax, x, y, data, "viridis", shading="flat")
        alpha = mesh.get_facecolor()[:, 3]
        assert alpha[0] == 0.0, f"NaN cell should be transparent, got alpha={alpha[0]}"

    def test_array_cleared_so_cmap_norm_do_not_override_facecolor(self, ax, xy):
        """`set_array(None)` is applied so the mesh renders the explicit facecolor."""
        x, y = xy
        data = np.array([[0.0, 1.0], [0.5, 0.25]])
        mesh = alpha_scaled_mesh(ax, x, y, data, "viridis", shading="flat")
        assert mesh.get_array() is None, "mesh array should be cleared after colouring"

    def test_non_2d_data_raises(self, ax, xy):
        """A 1D `data` array raises `ValueError`."""
        x, y = xy
        with pytest.raises(ValueError, match="2-dimensional"):
            alpha_scaled_mesh(ax, x, y, np.array([0.0, 1.0]), "viridis")

    def test_default_shading_is_auto(self, ax):
        """With no explicit `shading`, same-shape x/y/data does not raise."""
        x, y = np.meshgrid(np.arange(2), np.arange(2))
        data = np.array([[0.0, 1.0], [0.5, 0.25]])
        mesh = alpha_scaled_mesh(ax, x, y, data, "viridis")
        assert mesh in ax.collections, "default shading should still produce a mesh"


class TestApplyDataStyle:
    """Tests for `apply_data_style` and the `DATA_STYLES` registry."""

    @pytest.fixture
    def ax(self):
        """A fresh Axes on the Agg backend, closed after the test."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_cams_preset_has_both_layers(self):
        """The registered 'cams' preset defines exactly organic_matter and dust."""
        assert set(DATA_STYLES["cams"]) == {"organic_matter", "dust"}, (
            f"unexpected cams layers: {set(DATA_STYLES['cams'])}"
        )

    def test_draws_one_image_per_layer(self, ax):
        """Each key in `layers` produces one returned `AxesImage`, drawn on `ax`."""
        layers = {"dust": np.array([[0.0, 1.0]]), "organic_matter": np.array([[0.2, 0.8]])}
        images = apply_data_style(ax, layers)
        assert set(images) == {"dust", "organic_matter"}, f"unexpected keys: {set(images)}"
        for img in images.values():
            assert isinstance(img, AxesImage), f"expected AxesImage, got {type(img)}"
            assert img in ax.images, "image should be drawn on ax"

    def test_uses_the_layer_specific_colormap(self, ax):
        """Each layer is drawn with its own DATA_STYLES colormap, not a shared one."""
        images = apply_data_style(ax, {"dust": np.array([[0.0, 1.0]])})
        top_rgb = images["dust"].get_array()[0, 1, :3]
        expected = CAMS_COLORMAPS["dust"](1.0)[:3]
        np.testing.assert_allclose(
            top_rgb, expected, atol=1e-6, err_msg="dust layer used the wrong colormap"
        )

    def test_legend_true_attaches_one_swatch_per_layer(self, ax):
        """`legend=True` (the default) attaches one swatch legend per layer."""
        apply_data_style(ax, {"dust": np.array([[0.0, 1.0]]), "organic_matter": np.array([[0.0, 1.0]])})
        assert len(ax.child_axes) == 2, f"expected 2 swatch legends, got {len(ax.child_axes)}"

    def test_legend_false_attaches_no_swatch(self, ax):
        """`legend=False` draws the layers without any swatch legend."""
        apply_data_style(ax, {"dust": np.array([[0.0, 1.0]])}, legend=False)
        assert ax.child_axes == [], f"expected no swatch legends, got {ax.child_axes}"

    def test_partial_layer_subset_is_allowed(self, ax):
        """Passing only one of the preset's layers draws just that one."""
        images = apply_data_style(ax, {"dust": np.array([[0.0, 1.0]])})
        assert list(images) == ["dust"], f"expected only 'dust', got {list(images)}"

    def test_unknown_style_raises_key_error(self, ax):
        """An unregistered `style` name raises `KeyError` before drawing anything."""
        with pytest.raises(KeyError, match="Unknown data style"):
            apply_data_style(ax, {"dust": np.array([[0.0, 1.0]])}, style="not-a-style")
        assert len(ax.images) == 0, "nothing should be drawn when style is invalid"

    def test_unknown_layer_name_raises_key_error(self, ax):
        """A layer name the style doesn't define raises `KeyError`, nothing drawn."""
        with pytest.raises(KeyError, match="smoke"):
            apply_data_style(ax, {"smoke": np.array([[0.0, 1.0]])})
        assert len(ax.images) == 0, "nothing should be drawn when a layer is unknown"

    def test_explicit_legend_bounds_are_used(self, ax):
        """Explicit `legend_bounds` override the auto-stacked default position."""
        apply_data_style(
            ax,
            {"dust": np.array([[0.0, 1.0]])},
            legend_bounds=[(0.5, 0.5, 0.2, 0.05)],
        )
        assert ax.child_axes[0].get_position().bounds is not None, (
            "swatch should have a position derived from the explicit bounds"
        )

    def test_forwards_alpha_scaled_image_kwargs(self, ax):
        """Extra kwargs (e.g. `zorder`) reach the underlying `alpha_scaled_image`."""
        images = apply_data_style(ax, {"dust": np.array([[0.0, 1.0]])}, zorder=5)
        assert images["dust"].get_zorder() == 5, "zorder not forwarded to alpha_scaled_image"

    def test_x_y_dispatches_to_alpha_scaled_mesh(self, ax):
        """Passing `x`/`y` renders every layer as a `QuadMesh`, not an `AxesImage`."""
        x, y = np.meshgrid(np.arange(3), np.arange(3))
        images = apply_data_style(
            ax,
            {"dust": np.array([[0.0, 1.0], [0.5, 0.25]])},
            x=x,
            y=y,
            shading="flat",
        )
        assert isinstance(images["dust"], QuadMesh), (
            f"expected QuadMesh with x/y given, got {type(images['dust'])}"
        )

    def test_without_x_y_uses_alpha_scaled_image(self, ax):
        """With no `x`/`y`, layers render as `AxesImage` (the default path)."""
        images = apply_data_style(ax, {"dust": np.array([[0.0, 1.0]])})
        assert isinstance(images["dust"], AxesImage), (
            f"expected AxesImage without x/y, got {type(images['dust'])}"
        )

    @pytest.mark.parametrize("kwargs", [{"x": [[0.0, 1.0]]}, {"y": [[0.0, 1.0]]}])
    def test_only_one_of_x_y_raises(self, ax, kwargs):
        """Passing only `x` or only `y` raises `ValueError`, not a silent fallback.

        Args:
            kwargs: Either `{"x": ...}` or `{"y": ...}` alone.

        Test scenario:
            A caller who mis-destructures apply_projection_style's 3-tuple
            (e.g. passing only `x`) must get a clear error instead of
            silently falling back to the flat imshow path.
        """
        with pytest.raises(ValueError, match="x and y must be given together"):
            apply_data_style(ax, {"dust": np.array([[0.0, 1.0]])}, **kwargs)


class TestCreateColors:
    def test_create_from_hex(self):
        """test_create_colors_object."""
        hex_number = "ff0000"
        color = Colors(hex_number)
        assert color._color_value == [hex_number]

    def test_create_from_rgb(self):
        """test_create_colors_object."""
        rgb_color = (128, 51, 204)
        color = Colors(rgb_color)
        assert color._color_value == [rgb_color]

    def test_create_from_image(self, color_ramp_image: str):
        colors = Colors.create_from_image(color_ramp_image)
        assert isinstance(colors.color_value, list)
        assert len(colors.color_value) == 2713
        with pytest.raises(FileNotFoundError):
            Colors.create_from_image("color_ramp_image")

    def test_create_from_image_accepts_pathlib_path(self, color_ramp_image: str):
        """`create_from_image` accepts a `pathlib.Path`, not just `str` (issue #180)."""
        colors = Colors.create_from_image(Path(color_ramp_image))
        assert isinstance(colors.color_value, list)
        assert len(colors.color_value) == 2713

    def test_create_from_image_missing_pathlib_path_raises(self, tmp_path):
        """A missing `pathlib.Path` raises `FileNotFoundError` (widened type, error branch)."""
        with pytest.raises(FileNotFoundError):
            Colors.create_from_image(tmp_path / "does-not-exist.png")

    def test_raise_error(self, color_ramp_image: str):
        with pytest.raises(ValueError):
            Colors(11)


class TestColorRamp:
    def test_create_color_ramp(self, color_ramp_image: str):
        colors = Colors.create_from_image(color_ramp_image)
        color_ramp = colors.get_color_map()
        assert isinstance(color_ramp, LinearSegmentedColormap)


def test_get_type():
    """test_create_colors_object."""
    mixed_color = [(128, 51, 204), "#23a9dd", (0.5, 0.2, 0.8)]
    color = Colors(mixed_color)
    color_types = color.get_type()
    assert color_types == ["rgb", "hex", "rgb-normalized"]


def test_is_valid_rgb_norm_255():
    """test_create_colors_object."""
    rgb_color = (128, 51, 204)
    color = Colors(rgb_color)
    assert color._is_valid_rgb_255(rgb_color) is True
    rgb_color = (0.5, 0.2, 0.8)
    color = Colors(rgb_color)
    assert color._is_valid_rgb_norm(rgb_color) is True


def test_is_valid_rgb():
    """test_create_colors_object."""
    rgb_color = [(128, 51, 204), (0.5, 0.2, 0.8)]
    color = Colors(rgb_color)
    assert all(color.is_valid_rgb())


def test_is_valid_hex():
    """test_create_colors_object."""
    hex_number = ["ff0000", "#23a9dd", (128, 51, 204), (0.5, 0.2, 0.8)]
    color = Colors(hex_number)
    valid = color.is_valid_hex()
    assert valid == [False, True, False, False]


def test_to_rgb():
    """test_create_colors_object."""
    hex_number = ["#ff0000", "#23a9dd", (0.5, 0.2, 0.8), (35, 169, 221)]
    color = Colors(hex_number)
    rgb_scale_1 = color.to_rgb(normalized=True)
    assert rgb_scale_1 == [
        (1.0, 0.0, 0.0),
        (0.13725490196078433, 0.6627450980392157, 0.8666666666666667),
        (0.5, 0.2, 0.8),
        (0.13725490196078433, 0.6627450980392157, 0.8666666666666667),
    ]
    rgb_scale_255 = color.to_rgb(normalized=False)
    assert rgb_scale_255 == [
        (255, 0, 0),
        (35, 169, 221),
        (127, 51, 204),
        (35, 169, 221),
    ]


def test_to_hex():
    """test_create_colors_object."""
    mixed_color = [(128, 51, 204), "#23a9dd", (0.5, 0.2, 0.8)]
    color = Colors(mixed_color)
    hex_colors = color.to_hex()
    assert hex_colors == ["#8033cc", "#23a9dd", "#8033cc"]
