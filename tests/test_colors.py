from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from matplotlib.image import AxesImage

from cleopatra.colors import CAMS_COLORMAPS, Colors, alpha_scaled_image


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
