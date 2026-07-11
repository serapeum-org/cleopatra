import os
import re
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm, to_rgba
from matplotlib.figure import Figure
from matplotlib.text import Text
from PIL import Image

from cleopatra.array_glyph import (
    _COORD_DTYPE_MISMATCH,
    _COORD_SHAPE_MISMATCH,
    ArrayGlyph,
    FacetGrid,
)


class TestProperties:

    def test__str__(self):
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        array = ArrayGlyph(arr)
        assert isinstance(array.__str__(), str)


class TestCreateArray:
    def test_create_instance(self, arr: np.ndarray, no_data_value: float):
        array = ArrayGlyph(arr, exclude_value=[no_data_value])
        assert isinstance(array.arr, np.ndarray)
        # check if the first element is masked
        assert array.arr.mask[0, 0]
        assert array.num_domain_cells == 89
        assert array.vmin == 0
        assert array.vmax == 88

    def test_no_elem_is_deprecated_alias(self, arr: np.ndarray, no_data_value: float):
        """The legacy `no_elem` attribute still works but warns."""
        array = ArrayGlyph(arr, exclude_value=[no_data_value])
        with pytest.warns(DeprecationWarning, match="num_domain_cells"):
            value = array.no_elem
        assert value == array.num_domain_cells == 89


class TestRGB:
    def test_plot_rgb(self, sentinel_2: np.ndarray):
        extent = [
            34.626902783650785,
            34.654007151597256,
            31.82337186561403,
            31.8504762335605,
        ]
        array = ArrayGlyph(
            sentinel_2, rgb=[3, 2, 1], cutoff=[0.3, 0.3, 0.3], extent=extent
        )
        fig, ax = array.plot(title="Flow Accumulation")
        im = ax.get_images()[0]
        assert im.get_extent() == [extent[0], extent[2], extent[1], extent[3]]
        assert isinstance(fig, Figure)


class TestPlotArray:
    def test_plot_numpy_array(
        self,
        arr: np.ndarray,
        no_data_value: float,
    ):
        array = ArrayGlyph(arr, exclude_value=[no_data_value])
        fig, ax = array.plot(title="Flow Accumulation")
        assert isinstance(fig, Figure)

    def test_give_fig_ax(
        self,
        arr: np.ndarray,
        no_data_value: float,
    ):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot()
        array = ArrayGlyph(arr, exclude_value=[no_data_value], fig=fig, ax=ax)
        fig, ax = array.plot(title="Flow Accumulation")
        assert isinstance(fig, Figure)

    def test_plot_numpy_array_with_extent(
        self,
        arr: np.ndarray,
        no_data_value: float,
    ):
        extent = [
            -75.60441003848668,
            4.235054115032001,
            -75.09878783366909,
            4.704560448076901,
        ]
        array = ArrayGlyph(arr, exclude_value=[no_data_value], extent=extent)
        fig, ax = array.plot(title="Flow Accumulation")
        assert isinstance(fig, Figure)

    def test_plot_array_color_scale_1(
        self,
        arr: np.ndarray,
        no_data_value: float,
        cmap: str,
        color_scale: list,
        ticks_spacing: int,
    ):
        array = ArrayGlyph(arr, exclude_value=[no_data_value])
        fig, ax = array.plot(
            color_scale=color_scale[0], cmap=cmap, ticks_spacing=ticks_spacing
        )
        assert isinstance(fig, Figure)

    def test_plot_array_color_scale_2(
        self,
        arr: np.ndarray,
        no_data_value: float,
        cmap: str,
        color_scale_2_gamma: float,
        color_scale: list,
        ticks_spacing: int,
    ):
        array = ArrayGlyph(arr, exclude_value=[no_data_value])
        fig, ax = array.plot(
            color_scale=color_scale[1],
            cmap=cmap,
            gamma=color_scale_2_gamma,
            ticks_spacing=ticks_spacing,
        )
        assert isinstance(fig, Figure)

    def test_plot_array_color_scale_3(
        self,
        arr: np.ndarray,
        no_data_value: float,
        cmap: str,
        color_scale: list,
        ticks_spacing: int,
        color_scale_3_linscale: float,
        color_scale_3_linthresh: float,
    ):
        array = ArrayGlyph(arr, exclude_value=[no_data_value])
        fig, ax = array.plot(
            color_scale=color_scale[2],
            line_scale=color_scale_3_linscale,
            line_threshold=color_scale_3_linthresh,
            cmap=cmap,
            ticks_spacing=ticks_spacing,
        )

        assert isinstance(fig, Figure)

    def test_plot_array_color_scale_4(
        self,
        arr: np.ndarray,
        no_data_value: float,
        cmap: str,
        color_scale: list,
        ticks_spacing: int,
    ):
        array = ArrayGlyph(arr, exclude_value=[no_data_value])
        fig, ax = array.plot(color_scale=color_scale[3], cmap=cmap, ticks_spacing=5)

        assert isinstance(fig, Figure)

    def test_plot_array_color_scale_4_with_bounds(
        self,
        rhine_dem_arr: np.ndarray,
        cmap: str,
        color_scale: list,
        ticks_spacing: int,
        bounds: list,
        rhine_no_data_val: float,
    ):
        array = ArrayGlyph(rhine_dem_arr, exclude_value=[rhine_no_data_val])
        fig, ax = array.plot(
            color_scale=color_scale[3],
            cmap=cmap,
            ticks_spacing=ticks_spacing,
            bounds=bounds,
        )

        assert isinstance(fig, Figure)

    def test_plot_array_color_scale_5(
        self,
        arr: np.ndarray,
        no_data_value: float,
        cmap: str,
        color_scale: list,
        ticks_spacing: int,
        midpoint: int,
    ):
        array = ArrayGlyph(arr, exclude_value=[no_data_value])
        fig, ax = array.plot(
            color_scale=color_scale[4],
            midpoint=midpoint,
            cmap=cmap,
            ticks_spacing=ticks_spacing,
        )

        assert isinstance(fig, Figure)

    def test_plot_array_display_cell_values(
        self,
        arr: np.ndarray,
        no_data_value: float,
        ticks_spacing: int,
        display_cell_value: bool,
        num_size,
        background_color_threshold,
    ):
        array = ArrayGlyph(arr, exclude_value=[no_data_value])
        fig, ax = array.plot(
            display_cell_value=display_cell_value,
            num_size=num_size,
            background_color_threshold=background_color_threshold,
            ticks_spacing=ticks_spacing,
        )

        assert isinstance(fig, Figure)

    def test_plot_array_with_points(
        self,
        arr: np.ndarray,
        no_data_value: float,
        display_cell_value: bool,
        points,
        num_size,
        background_color_threshold,
        ticks_spacing: int,
        id_size: int,
        id_color: str,
        point_size: int,
        gauge_color: str,
    ):
        array = ArrayGlyph(arr, exclude_value=[no_data_value])
        fig, ax = array.plot(
            points=points,
            point_color=gauge_color,
            point_size=point_size,
            id_color=id_color,
            id_size=id_size,
            display_cell_value=display_cell_value,
            num_size=num_size,
            background_color_threshold=background_color_threshold,
            ticks_spacing=ticks_spacing,
        )

        assert isinstance(fig, Figure)


class TestAnimate:
    def test_numpy_array(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        array = ArrayGlyph(coello_data, exclude_value=[no_data_value])
        anim_obj = array.animate(animate_time_list, title="Flow Accumulation")
        assert isinstance(anim_obj, FuncAnimation)

    def test_save_animation_accepts_pathlib_path(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
        tmp_path,
    ):
        """`ArrayGlyph.save_animation` accepts a `pathlib.Path` directly (issue #180)."""
        array = ArrayGlyph(coello_data, exclude_value=[no_data_value])
        array.animate(animate_time_list, title="Flow Accumulation")
        out = tmp_path / "flow.gif"  # Path, not str
        array.save_animation(out, fps=2)
        assert out.exists(), "GIF was not written from a pathlib.Path"
        assert out.read_bytes()[:6] in (b"GIF87a", b"GIF89a")

    def test_save_animation_gif(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        video_format = "gif"
        path = f"tests/data/animation.{video_format}"
        if os.path.exists(path):
            os.remove(path)

        array = ArrayGlyph(coello_data, exclude_value=[no_data_value])
        anim = array.animate(animate_time_list, title="Flow Accumulation")
        array.save_animation(path, fps=2)
        # assert Path(path).exists()
        # os.remove(path)

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
    def test_save_animation_avi(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        video_format = "avi"
        path = f"tests/data/animation.{video_format}"
        if os.path.exists(path):
            os.remove(path)

        array = ArrayGlyph(coello_data, exclude_value=[no_data_value])
        anim = array.animate(animate_time_list, title="Flow Accumulation")
        array.save_animation(path, fps=2)
        # assert Path(path).exists()
        # os.remove(path)

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
    def test_save_animation_mp4(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        video_format = "mp4"
        path = f"tests/data/animation.{video_format}"
        if os.path.exists(path):
            os.remove(path)

        array = ArrayGlyph(coello_data, exclude_value=[no_data_value])
        anim = array.animate(animate_time_list, title="Flow Accumulation")
        array.save_animation(path, fps=2)
        # assert Path(path).exists()
        # os.remove(path)

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
    def test_save_animation_mov(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        video_format = "mov"
        path = f"tests/data/animation.{video_format}"
        if os.path.exists(path):
            os.remove(path)

        array = ArrayGlyph(coello_data, exclude_value=[no_data_value])
        anim = array.animate(animate_time_list, title="Flow Accumulation")
        array.save_animation(path, fps=2)
        # assert Path(path).exists()
        # os.remove(path)


@pytest.mark.plot
class TestAnimateRGB:
    """RGB / true-colour animation support in `ArrayGlyph.animate` (issue #168)."""

    _LABELS = ["t0", "t1", "t2"]

    @staticmethod
    def _rgb_stack(channels: int = 3, n: int = 3, h: int = 8, w: int = 8) -> np.ndarray:
        """Build a deterministic 4-D true-colour stack in the 0-1 range.

        Args:
            channels: Trailing channel count (3 for RGB, 4 for RGBA).
            n: Number of time steps.
            h: Frame height.
            w: Frame width.

        Returns:
            np.ndarray: A `(n, h, w, channels)` float array with values in [0, 1].
        """
        size = n * h * w * channels
        return np.linspace(0.0, 1.0, size).reshape(n, h, w, channels)

    @pytest.mark.parametrize("channels", [3, 4])
    def test_rgb_stack_returns_animation_without_colorbar(self, channels):
        """A 4-D RGB/RGBA stack animates with no colorbar.

        Args:
            channels: Trailing channel count (3 = RGB, 4 = RGBA).

        Test scenario:
            `ArrayGlyph(stack).animate(labels)` on a
            `(time, rows, cols, 3|4)` stack returns a `FuncAnimation`,
            sets the first-frame artist on `self.im`, and leaves
            `self.cbar` as None (true colour needs no colorbar).
        """
        glyph = ArrayGlyph(self._rgb_stack(channels=channels))
        anim = glyph.animate(self._LABELS)
        assert isinstance(
            anim, FuncAnimation
        ), f"expected FuncAnimation, got {type(anim).__name__}"
        assert (
            glyph.cbar is None
        ), f"RGB animation must have no colorbar, got {glyph.cbar}"
        assert glyph.im is not None, "first-frame artist should be stored on self.im"

    def test_rgb_animation_renders_to_gif(self, tmp_path):
        """Saving an RGB animation renders every frame to a file.

        Args:
            tmp_path: pytest temporary directory fixture.

        Test scenario:
            Saving the returned animation produces a non-empty GIF,
            proving the per-frame `set_data` update path runs for RGB.
        """
        glyph = ArrayGlyph(self._rgb_stack(), title="RGB")
        glyph.animate(self._LABELS)
        out = tmp_path / "rgb_anim.gif"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glyph.save_animation(str(out), fps=2)
        assert out.exists(), "RGB animation file was not written"
        assert out.stat().st_size > 0, "RGB animation file is empty"

    def test_data_getter_returns_rgb_frames(self):
        """The lazy `data_getter` path accepts RGB frames over a 2-D template.

        Test scenario:
            A 2-D shape template plus a callback returning `(h, w, 3)`
            frames animates as true colour, with no colorbar.
        """
        template = np.zeros((8, 8))
        glyph = ArrayGlyph(template)

        def get_rgb(i):
            return np.full((8, 8, 3), i / 3.0)

        anim = glyph.animate(self._LABELS, data_getter=get_rgb)
        assert isinstance(
            anim, FuncAnimation
        ), f"expected FuncAnimation, got {type(anim).__name__}"
        assert (
            glyph.cbar is None
        ), f"RGB data_getter animation must have no colorbar, got {glyph.cbar}"

    def test_data_getter_rgb_renders_to_gif(self, tmp_path):
        """Playing a lazy RGB animation fetches and renders every frame.

        Args:
            tmp_path: pytest temporary directory fixture.

        Test scenario:
            Saving a `data_getter`-backed RGB animation exercises the
            per-frame fetch + spatial-shape revalidation for true colour
            and writes a non-empty GIF.
        """
        glyph = ArrayGlyph(np.zeros((8, 8)))

        def get_rgb(i):
            return np.full((8, 8, 3), (i + 1) / 4.0)

        glyph.animate(self._LABELS, data_getter=get_rgb)
        out = tmp_path / "rgb_lazy.gif"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glyph.save_animation(str(out), fps=2)
        assert out.exists(), "lazy RGB animation file was not written"
        assert out.stat().st_size > 0, "lazy RGB animation file is empty"

    def test_data_getter_bad_spatial_shape_raises(self):
        """A `data_getter` whose spatial dims mismatch the template raises.

        Test scenario:
            A `(5, 5, 3)` frame against an `(8, 8)` template must raise
            `ValueError` mentioning that the spatial dims do not match.
        """
        glyph = ArrayGlyph(np.zeros((8, 8)))
        with pytest.raises(ValueError, match="do not match") as exc:
            glyph.animate(self._LABELS, data_getter=lambda i: np.zeros((5, 5, 3)))
        assert "do not match" in str(
            exc.value
        ), f"unexpected error message: {exc.value}"

    def test_display_cell_value_ignored_for_rgb(self):
        """`display_cell_value=True` is silently skipped for RGB frames.

        Test scenario:
            Per-cell annotation needs a scalar field; on an RGB stack the
            option must be ignored without raising, still returning a
            `FuncAnimation` with no colorbar.
        """
        glyph = ArrayGlyph(self._rgb_stack())
        anim = glyph.animate(self._LABELS, display_cell_value=True)
        assert isinstance(
            anim, FuncAnimation
        ), f"expected FuncAnimation, got {type(anim).__name__}"
        assert glyph.cbar is None, "RGB animation must have no colorbar"

    def test_single_band_3d_path_unchanged(self):
        """The 3-D single-band path still draws a colorbar (no regression).

        Test scenario:
            A `(time, rows, cols)` stack animates as a colormapped field
            and keeps its colorbar — the RGB branch must not affect it.
        """
        single_band = np.linspace(0.0, 1.0, 3 * 8 * 8).reshape(3, 8, 8)
        glyph = ArrayGlyph(single_band)
        anim = glyph.animate(self._LABELS)
        assert isinstance(
            anim, FuncAnimation
        ), f"expected FuncAnimation, got {type(anim).__name__}"
        assert glyph.cbar is not None, "single-band animation should keep its colorbar"

    def test_2d_without_data_getter_raises_naming_both_dims(self):
        """A 2-D array with no `data_getter` errors, naming 3-D and 4-D.

        Test scenario:
            With no time axis and no callback, `animate` must raise
            `ValueError` whose message names both the 3-D and 4-D
            accepted shapes.
        """
        glyph = ArrayGlyph(np.zeros((8, 8)))
        with pytest.raises(ValueError) as exc:
            glyph.animate(self._LABELS)
        msg = str(exc.value)
        assert (
            "3-D" in msg and "4-D" in msg
        ), f"message should name both 3-D and 4-D, got: {msg}"


class TestNoImplicitShow:
    """Tests that `plot()` and `animate()` do not force an interactive display.

    The branch removed the internal ``plt.show()`` calls so the methods return the
    Figure/Axes (or animation) for the caller to display or save. These tests pin that
    contract by failing if ``plt.show`` is ever invoked.
    """

    def test_plot_does_not_call_plt_show(
        self,
        monkeypatch: pytest.MonkeyPatch,
        arr: np.ndarray,
        no_data_value: float,
    ):
        """Test that `ArrayGlyph.plot()` returns (fig, ax) without calling `plt.show`.

        Args:
            monkeypatch: Pytest fixture used to replace ``matplotlib.pyplot.show``.
            arr: Sample 2D array fixture.
            no_data_value: Value masked out of the array fixture.

        Test scenario:
            Patch ``plt.show`` with a counter. After ``plot()`` the counter must be 0,
            and the method must still return a Figure and an Axes.
        """
        calls = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
        array = ArrayGlyph(arr, exclude_value=[no_data_value])
        fig, ax = array.plot(title="Flow Accumulation")
        assert (
            calls == []
        ), f"plot() should not call plt.show(); was called {len(calls)} time(s)"
        assert isinstance(
            fig, Figure
        ), f"plot() should return a Figure, got {type(fig)}"
        assert ax is not None, "plot() should return an Axes alongside the Figure"

    def test_animate_does_not_call_plt_show(
        self,
        monkeypatch: pytest.MonkeyPatch,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        """Test that `ArrayGlyph.animate()` returns an animation without calling `plt.show`.

        Args:
            monkeypatch: Pytest fixture used to replace ``matplotlib.pyplot.show``.
            coello_data: Sample 3D (time-stacked) array fixture.
            animate_time_list: Per-frame time labels fixture.
            no_data_value: Value masked out of the array fixture.

        Test scenario:
            Patch ``plt.show`` with a counter. After ``animate()`` the counter must be 0,
            and the method must still return a ``FuncAnimation``.
        """
        calls = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
        array = ArrayGlyph(coello_data, exclude_value=[no_data_value])
        anim = array.animate(animate_time_list, title="Flow Accumulation")
        assert (
            calls == []
        ), f"animate() should not call plt.show(); was called {len(calls)} time(s)"
        assert isinstance(
            anim, FuncAnimation
        ), f"animate() should return a FuncAnimation, got {type(anim)}"


def test_scale_percentile():
    arr = np.array(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ]
    )
    array = ArrayGlyph(arr)

    scaled_arr = np.array(
        [
            [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]],
            [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]],
            [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]],
        ]
    )
    np.testing.assert_array_almost_equal(array.scale_percentile(arr), scaled_arr)


def test_apply_color_map():
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    my_glyph = ArrayGlyph(arr)
    cmap = "viridis"
    colored_arr = my_glyph.apply_colormap(cmap)
    assert colored_arr.shape == (4, 4, 3)
    assert colored_arr.dtype == "uint8"


class TestScaleToRGB:
    def test_scale_to_rgb(self):
        arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        my_glyph = ArrayGlyph(arr)

        rgb_arr = my_glyph.scale_to_rgb()
        assert rgb_arr.shape == (4, 4)
        assert rgb_arr.dtype == "uint8"

        rgb_arr = my_glyph.scale_to_rgb(arr=arr)
        assert rgb_arr.shape == (4, 4)
        assert rgb_arr.dtype == "uint8"


class TestToImage:
    def test_int64_arr(self):
        arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        my_glyph = ArrayGlyph(arr)

        image = my_glyph.to_image()
        assert isinstance(image, Image.Image)

        # test if you provide the arr
        image = my_glyph.to_image(arr=arr)
        assert isinstance(image, Image.Image)

    def test_uint_arr(self):
        arr = np.random.randint(0, 255, size=(4, 4)).astype(np.uint8)
        my_glyph = ArrayGlyph(arr)

        image = my_glyph.to_image()
        assert isinstance(image, Image.Image)


def test_adjust_ticks():
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    extent = [
        34.626902783650785,
        34.654007151597256,
        31.82337186561403,
        31.8504762335605,
    ]
    my_glyph = ArrayGlyph(arr, extent=extent)
    fig, ax = my_glyph.plot()
    my_glyph.adjust_ticks(axis="x", multiply_value=100)
    ax = my_glyph.ax
    values = [val.get_text() for val in ax.xaxis.get_ticklabels()]
    assert values == ["3100", "3200", "3300", "3400", "3500"]


@pytest.mark.plot
class TestPlotKindDispatch:
    """Tests for `ArrayGlyph.plot(kind=...)` dispatch.

    Covers the `kind` enum introduced in CLEO-1:
    `"auto" | "imshow" | "pcolormesh" | "contour" | "contourf"`.
    """

    @staticmethod
    def _sample_arr() -> np.ndarray:
        """Return a small 2-D array suitable for every kind."""
        return np.arange(25, dtype=float).reshape(5, 5)

    def test_default_kind_is_backward_compatible(self):
        """`plot()` with no kind kwarg still returns (fig, ax) with an image."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot()
        assert isinstance(fig, Figure)
        assert len(ax.get_images()) >= 1

    def test_kind_imshow(self):
        """`kind="imshow"` renders via `ax.imshow`/`matshow`."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="imshow")
        assert isinstance(fig, Figure)
        assert len(ax.get_images()) >= 1

    def test_kind_pcolormesh(self):
        """`kind="pcolormesh"` runs and adds a `QuadMesh` to `ax`."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="pcolormesh")
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1

    def test_kind_contour(self):
        """`kind="contour"` runs and adds contour collections to `ax`."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="contour")
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1

    def test_kind_contourf(self):
        """`kind="contourf"` runs and adds filled contour collections."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="contourf")
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1

    def test_invalid_kind_raises(self):
        """`kind="bogus"` raises `ValueError` listing the valid kinds."""
        glyph = ArrayGlyph(self._sample_arr())
        with pytest.raises(ValueError) as exc:
            glyph.plot(kind="bogus")
        msg = str(exc.value)
        # The message must mention every supported kind so users can fix
        # the typo without consulting the docs.
        for kind in ("auto", "imshow", "pcolormesh", "contour", "contourf"):
            assert kind in msg

    def test_rgb_with_non_imshow_kind_raises(self):
        """RGB compositing is imshow-only — other kinds must raise."""
        rgb_arr = np.random.randint(0, 255, size=(3, 8, 8)).astype(np.float32)
        glyph = ArrayGlyph(rgb_arr, rgb=[0, 1, 2])
        with pytest.raises(ValueError, match="RGB"):
            glyph.plot(kind="pcolormesh")

    def test_color_scale_linear_with_auto(self):
        """Linear color_scale works under the default kind."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(color_scale="linear")
        assert isinstance(fig, Figure)

    def test_color_scale_power_with_auto(self):
        """Power color_scale works under the default kind."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(color_scale="power")
        assert isinstance(fig, Figure)

    def test_color_scale_enum_member_accepted(self):
        """`plot(color_scale=ColorScale.POWER)` works (enum member, not str)."""
        from cleopatra.styles import ColorScale

        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(color_scale=ColorScale.POWER)
        assert isinstance(fig, Figure)

    def test_invalid_color_scale_int_raises_valueerror(self):
        """`plot(color_scale=1)` raises `ValueError` (not `AttributeError`)."""
        glyph = ArrayGlyph(self._sample_arr())
        with pytest.raises(ValueError, match="Invalid color_scale"):
            glyph.plot(color_scale=1)

    def test_color_scale_linear_with_pcolormesh(self):
        """Linear color_scale works under kind=pcolormesh."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="pcolormesh", color_scale="linear")
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1

    def test_color_scale_power_with_pcolormesh(self):
        """Power color_scale works under kind=pcolormesh."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="pcolormesh", color_scale="power")
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1

    def test_imshow_with_display_cell_value(self):
        """`kind="imshow", display_cell_value=True` still draws cell text."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="imshow", display_cell_value=True)
        assert isinstance(fig, Figure)
        # Each cell of the 5x5 grid yields one Text artist; allow >= 25
        # because matplotlib also creates axis label/title Text objects.
        cell_texts = [t for t in ax.texts]
        assert len(cell_texts) >= 25

    def test_contour_skips_cell_value_silently(self):
        """`display_cell_value=True` is skipped for `kind="contour"`."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="contour", display_cell_value=True)
        assert isinstance(fig, Figure)
        # No per-cell text annotations should have been emitted.
        assert len(ax.texts) == 0


@pytest.mark.plot
class TestColourKwargs:
    """Tests for the xarray-aligned colour kwargs introduced in CLEO-3.

    Covers `robust`, `center`, `levels`, `extend`, and
    `cbar_kwargs` on `ArrayGlyph`. Each kwarg is tested both in
    isolation and (where relevant) in combination with the existing
    `color_scale` enum and the `kind=` dispatch from CLEO-1.
    """

    @staticmethod
    def _arr_with_outliers() -> np.ndarray:
        """Build a 10x10 array of normal values plus extreme outliers.

        The bulk of the values sit in `[0, 1)`; two cells hold large
        negative and positive outliers so `robust=True` has something
        to clip.
        """
        rng = np.random.default_rng(seed=0)
        body = rng.random(98)
        full = np.concatenate([body, [-1000.0, 1000.0]])
        return full.reshape(10, 10)

    def test_robust_clips_vmin_vmax(self):
        """`robust=True` uses the 2nd/98th percentile, ignoring outliers."""
        arr = self._arr_with_outliers()
        glyph = ArrayGlyph(arr, robust=True)
        full_min = float(np.nanmin(arr))
        full_max = float(np.nanmax(arr))
        # The robust limits must be tighter than the data range — that
        # is the whole point of clipping the 2nd/98th percentile.
        assert glyph.vmin > full_min
        assert glyph.vmax < full_max
        # And much tighter than the outliers themselves (they sit at
        # +/-1000; the bulk lives in [0, 1)).
        assert abs(glyph.vmin) < 5.0
        assert abs(glyph.vmax) < 5.0

    def test_robust_does_not_override_explicit_vmin_vmax(self):
        """An explicit `vmin`/`vmax` always wins over `robust=True`."""
        arr = self._arr_with_outliers()
        glyph = ArrayGlyph(arr, robust=True, vmin=0.0, vmax=1.0)
        assert glyph.vmin == 0.0
        assert glyph.vmax == 1.0

    def test_center_zero_makes_vmin_vmax_symmetric(self):
        """`center=0` symmetrises around zero using the larger half-range."""
        arr = np.linspace(-3.0, 7.0, 30).reshape(5, 6)
        glyph = ArrayGlyph(arr, center=0.0)
        assert glyph.vmin == pytest.approx(-7.0)
        assert glyph.vmax == pytest.approx(7.0)

    def test_center_switches_default_cmap_to_rdbu_r(self):
        """`center=...` without an explicit cmap selects `RdBu_r`."""
        arr = np.linspace(-2.0, 2.0, 16).reshape(4, 4)
        glyph = ArrayGlyph(arr, center=0.0)
        assert glyph.default_options["cmap"] == "RdBu_r"

    def test_center_does_not_override_explicit_cmap(self):
        """An explicit `cmap=` always wins over the diverging default."""
        arr = np.linspace(-2.0, 2.0, 16).reshape(4, 4)
        glyph = ArrayGlyph(arr, center=0.0, cmap="viridis")
        assert glyph.default_options["cmap"] == "viridis"

    def test_levels_int_creates_discretised_norm(self):
        """`levels=N` switches the linear norm to a `BoundaryNorm`."""
        arr = np.linspace(0.0, 1.0, 100).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        fig, ax = glyph.plot(kind="imshow", levels=5)
        assert isinstance(fig, Figure)
        images = ax.get_images()
        assert images, "expected at least one AxesImage from imshow"
        norm = images[0].norm
        assert isinstance(
            norm, BoundaryNorm
        ), f"levels=int should switch to BoundaryNorm, got {type(norm)}"
        assert len(norm.boundaries) == 5

    def test_levels_list_uses_explicit_edges(self):
        """`levels=[...]` is forwarded as explicit bin edges."""
        arr = np.linspace(0.0, 1.0, 100).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        edges = [0.0, 0.25, 0.5, 0.75, 1.0]
        fig, ax = glyph.plot(kind="imshow", levels=edges)
        assert isinstance(fig, Figure)
        norm = ax.get_images()[0].norm
        assert isinstance(norm, BoundaryNorm)
        np.testing.assert_array_almost_equal(norm.boundaries, edges)

    def test_extend_both_when_levels_set_and_extend_none(self):
        """`levels=[...], extend=None` auto-resolves `extend` to `both`."""
        arr = np.linspace(0.0, 1.0, 100).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        edges = [0.0, 0.5, 1.0]
        fig, ax = glyph.plot(kind="imshow", levels=edges)
        # The cbar is attached to the image's mappable; resolve via
        # the glyph attribute we expose at create_color_bar time.
        cbar = glyph.cbar
        assert cbar.extend == "both"

    def test_extend_explicit_overrides_auto(self):
        """An explicit `extend="min"` is preserved despite `levels`."""
        arr = np.linspace(0.0, 1.0, 100).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        edges = [0.0, 0.5, 1.0]
        fig, ax = glyph.plot(kind="imshow", levels=edges, extend="min")
        assert glyph.cbar.extend == "min"

    def test_cbar_kwargs_merge_overrides_defaults(self):
        """`cbar_kwargs` merges over the defaults; user keys win."""
        arr = np.linspace(0.0, 1.0, 100).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        fig, ax = glyph.plot(
            kind="imshow",
            cbar_kwargs={"label": "Custom", "shrink": 0.5},
        )
        cbar = glyph.cbar
        # The label set via cbar_kwargs flows through `set_label` and
        # surfaces on the underlying `YAxis` label artist.
        assert cbar.ax.get_ylabel() == "Custom"
        # And the shrink override flows through to the figure colorbar
        # creation — verify by reading the colorbar axes bbox height
        # ratio relative to the parent.
        parent_h = ax.get_position().height
        cbar_h = cbar.ax.get_position().height
        assert cbar_h <= parent_h * 0.75, (
            "shrink=0.5 should produce a shorter colorbar than the "
            "default shrink=0.75"
        )

    def test_robust_combined_with_center(self):
        """`robust=True, center=0` clips outliers then symmetrises."""
        rng = np.random.default_rng(seed=1)
        body = rng.uniform(-10.0, 5.0, size=98)
        arr = np.concatenate([body, [-100.0, 50.0]]).reshape(10, 10)
        glyph = ArrayGlyph(arr, robust=True, center=0.0)
        # Symmetric around zero.
        assert glyph.vmin == pytest.approx(-glyph.vmax)
        # And much tighter than the full [-100, 100] symmetric range.
        assert abs(glyph.vmin) < 50.0
        assert abs(glyph.vmax) < 50.0

    def test_robust_works_with_kind_pcolormesh(self):
        """`robust=True, kind="pcolormesh"` renders a QuadMesh."""
        arr = self._arr_with_outliers()
        glyph = ArrayGlyph(arr, robust=True)
        fig, ax = glyph.plot(kind="pcolormesh")
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1


@pytest.mark.plot
class TestValidateExtend:
    """Tests for :py`ArrayGlyph._validate_extend` static method."""

    def test_none_is_accepted(self):
        """`extend=None` returns silently (auto-resolve at render time)."""
        result = ArrayGlyph._validate_extend(None)
        assert result is None, f"Expected None, got {result!r}"

    @pytest.mark.parametrize("value", ["neither", "both", "min", "max"])
    def test_allowed_values_accepted(self, value):
        """Each allowed string returns silently.

        Args:
            value: A valid `extend` keyword.
        """
        ArrayGlyph._validate_extend(value)

    @pytest.mark.parametrize("bogus", ["always", "MIN", "", "neither2", "extend"])
    def test_invalid_string_raises(self, bogus):
        """Any other string raises `ValueError` listing the allowed set.

        Args:
            bogus: An invalid extend value to reject.
        """
        with pytest.raises(ValueError, match="Invalid extend"):
            ArrayGlyph._validate_extend(bogus)

    def test_invalid_extend_in_constructor_raises(self):
        """Passing `extend='banana'` to `__init__` raises `ValueError`."""
        with pytest.raises(ValueError, match="Invalid extend"):
            ArrayGlyph(np.arange(9).reshape(3, 3), extend="banana")

    def test_invalid_extend_in_plot_raises(self):
        """Passing `extend='banana'` to `plot` raises `ValueError`."""
        glyph = ArrayGlyph(np.arange(9).reshape(3, 3))
        with pytest.raises(ValueError, match="Invalid extend"):
            glyph.plot(extend="banana")


@pytest.mark.plot
class TestRobustLimits:
    """Tests for :py`ArrayGlyph._robust_limits` static method."""

    def test_basic_array(self):
        """A plain array returns 2nd / 98th percentile floats."""
        rng = np.random.default_rng(1337)
        arr = rng.normal(size=10000)
        vmin_r, vmax_r = ArrayGlyph._robust_limits(arr)
        assert vmin_r > arr.min(), "Robust vmin should exceed full min"
        assert vmax_r < arr.max(), "Robust vmax should fall under full max"

    def test_all_nan_raises(self):
        """An all-NaN array surfaces as `ValueError`."""
        arr = np.full((5, 5), np.nan)
        with pytest.raises(ValueError, match="no finite values"):
            ArrayGlyph._robust_limits(arr)

    def test_all_inf_raises(self):
        """An all-Inf array (no finite cells) surfaces as `ValueError`."""
        arr = np.full((5, 5), np.inf)
        with pytest.raises(ValueError, match="no finite values"):
            ArrayGlyph._robust_limits(arr)

    def test_masked_array_uses_compressed_values(self):
        """Masked entries are excluded from the percentile computation."""
        import numpy.ma as ma

        data = np.array([1.0, 2.0, 3.0, 4.0, 1e6])
        mask = np.array([False, False, False, False, True])
        marr = ma.array(data, mask=mask)
        _, vmax_r = ArrayGlyph._robust_limits(marr)
        assert vmax_r < 1e6, "Outlier masked cell must be ignored"


@pytest.mark.plot
class TestCenterLimits:
    """Tests for :py`ArrayGlyph._center_limits` static method."""

    def test_symmetric_around_zero(self):
        """A skewed range becomes symmetric around zero using the larger half."""
        vmin, vmax = ArrayGlyph._center_limits(-3.0, 7.0, 0.0)
        assert vmin == pytest.approx(-7.0)
        assert vmax == pytest.approx(7.0)

    def test_symmetric_around_nonzero(self):
        """Centring on a nonzero value still uses the larger half-range."""
        vmin, vmax = ArrayGlyph._center_limits(0.0, 10.0, 4.0)
        half = max(abs(0.0 - 4.0), abs(10.0 - 4.0))
        assert vmin == pytest.approx(4.0 - half)
        assert vmax == pytest.approx(4.0 + half)

    def test_already_symmetric(self):
        """An already-symmetric range is preserved."""
        vmin, vmax = ArrayGlyph._center_limits(-5.0, 5.0, 0.0)
        assert vmin == pytest.approx(-5.0)
        assert vmax == pytest.approx(5.0)


class TestAllNaNColorLimits:
    """Constructing from an array with no finite values fails loudly (M2)."""

    def test_all_nan_array_raises(self):
        """`ArrayGlyph(all_nan)` raises `ValueError`, not a NaN colour range."""
        with pytest.raises(ValueError, match="no finite values"):
            ArrayGlyph(np.full((5, 5), np.nan))

    def test_fully_masked_array_raises(self):
        """A fully-masked array (every cell == exclude_value) raises too."""
        arr = np.full((4, 4), -9999.0)
        with pytest.raises(ValueError, match="no finite values"):
            ArrayGlyph(arr, exclude_value=[-9999.0])

    def test_robust_all_nan_raises(self):
        """The `robust=True` path also raises on an all-NaN array."""
        with pytest.raises(ValueError, match="no finite"):
            ArrayGlyph(np.full((5, 5), np.nan), robust=True)

    def test_explicit_vmin_vmax_override_makes_all_nan_usable(self):
        """Passing explicit `vmin`/`vmax` lets an all-NaN array through.

        This is what `facet` relies on for its 0-1 fallback on an all-NaN
        stack panel.
        """
        glyph = ArrayGlyph(np.full((5, 5), np.nan), vmin=0.0, vmax=1.0)
        assert glyph.vmin == 0.0
        assert glyph.vmax == 1.0


class TestPrepareArrayValidation:
    """Tests for `ArrayGlyph.prepare_array` and RGB constructor validation."""

    def test_too_few_bands_raises(self):
        """An RGB array with fewer than 3 bands raises `ValueError`."""
        arr = np.zeros((2, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="3 arrays"):
            ArrayGlyph(arr, rgb=[0, 1])

    def test_prepare_array_with_cutoff_only(self):
        """`cutoff` is applied via the surface-reflectance branch."""
        arr = (
            np.random.default_rng(0)
            .integers(0, 10000, size=(3, 5, 5))
            .astype(np.float32)
        )
        glyph = ArrayGlyph(np.zeros((1, 1)))
        result = glyph.prepare_array(
            arr, rgb=[0, 1, 2], surface_reflectance=10000, cutoff=[5000, 5000, 5000]
        )
        assert result.shape == (5, 5, 3)
        assert np.all((0.0 <= result) & (result <= 1.0))

    def test_prepare_array_no_normalisation(self):
        """No percentile and no surface_reflectance -> only reorder bands."""
        arr = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        glyph = ArrayGlyph(np.zeros((1, 1)))
        result = glyph.prepare_array(arr, rgb=[0, 1, 2])
        assert result.shape == (3, 3, 3)
        np.testing.assert_array_equal(result[..., 0], arr[0])

    def test_prepare_sentinel_rgb_no_cutoff(self):
        """`_prepare_sentinel_rgb` returns clipped data with no cutoff path."""
        arr = (
            np.random.default_rng(0)
            .integers(0, 10000, size=(3, 5, 5))
            .astype(np.float32)
        )
        glyph = ArrayGlyph(np.zeros((1, 1)))
        result = glyph.prepare_array(arr, rgb=[0, 1, 2], surface_reflectance=10000)
        assert result.shape == (5, 5, 3)
        assert np.all((0.0 <= result) & (result <= 1.0))


class TestPlotImGetCbarKwInvalidKind:
    """Tests for :py`ArrayGlyph._plot_im_get_cbar_kw` invalid-kind branch."""

    def test_invalid_kind_raises(self):
        """An unsupported kind in the helper raises `ValueError`.

        Test scenario:
            `ArrayGlyph.plot` validates the kind before calling
            `_plot_im_get_cbar_kw`; reaching the helper directly with
            a bogus kind exercises the defensive branch in the helper.
        """
        glyph = ArrayGlyph(np.arange(25, dtype=float).reshape(5, 5))
        fig, ax = glyph.create_figure_axes()
        try:
            glyph.default_options["vmin"] = 0.0
            glyph.default_options["vmax"] = 24.0
            glyph.default_options["ticks_spacing"] = 5.0
            ticks = glyph.get_ticks()
            with pytest.raises(ValueError, match="Invalid kind"):
                glyph._plot_im_get_cbar_kw(ax, glyph.arr, ticks, kind="banana")
        finally:
            plt.close(fig)


class TestArrSetter:
    """Tests for the `arr` property setter."""

    def test_setter_replaces_array(self):
        """Assigning to `arr` swaps out the internal masked array."""
        a = np.arange(9).reshape(3, 3)
        glyph = ArrayGlyph(a)
        new = np.zeros((4, 4))
        glyph.arr = new
        assert glyph.arr is new, "Setter should store the new array reference"


class TestExcludeValueProperty:
    """Tests for the `exclude_value` getter/setter."""

    def test_default_is_nan(self):
        """The default `exclude_value` is `np.nan`."""
        glyph = ArrayGlyph(np.arange(9).reshape(3, 3))
        # exclude_value comparison: nan is nan via identity
        assert glyph.exclude_value is np.nan or (
            isinstance(glyph.exclude_value, float) and np.isnan(glyph.exclude_value)
        )

    def test_setter_updates_value(self):
        """Setter updates the stored exclude value."""
        glyph = ArrayGlyph(np.arange(9).reshape(3, 3))
        glyph.exclude_value = -9999
        assert glyph.exclude_value == -9999


@pytest.mark.plot
class TestPlotRecomputeBranch:
    """Tests for the recompute-on-plot path in :py`ArrayGlyph.plot`."""

    def test_passing_robust_in_plot_recomputes_limits(self):
        """`plot(robust=True)` recomputes vmin/vmax from the percentile path."""
        rng = np.random.default_rng(1337)
        body = rng.random(98)
        arr = np.concatenate([body, [-1000.0, 1000.0]]).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        before_vmin = glyph.vmin
        glyph.plot(robust=True)
        assert (
            glyph.vmin > before_vmin
        ), "Robust pass on plot should clip the lower outlier"
        assert glyph.vmax < 1000.0, "Robust pass on plot should clip the upper outlier"

    def test_passing_center_in_plot_switches_cmap(self):
        """`plot(center=0)` selects `RdBu_r` when no explicit cmap is given."""
        arr = np.linspace(-3.0, 5.0, 25).reshape(5, 5)
        glyph = ArrayGlyph(arr)
        glyph.plot(center=0.0)
        assert glyph.default_options["cmap"] == "RdBu_r"

    def test_passing_center_in_plot_with_explicit_cmap_keeps_user_choice(self):
        """`plot(center=0, cmap='viridis')` keeps the explicit cmap."""
        arr = np.linspace(-3.0, 5.0, 25).reshape(5, 5)
        glyph = ArrayGlyph(arr)
        glyph.plot(center=0.0, cmap="viridis")
        assert glyph.default_options["cmap"] == "viridis"

    def test_passing_explicit_vmin_vmax_in_plot_overrides_constructor(self):
        """`plot(vmin=, vmax=)` overrides constructor-derived limits."""
        arr = np.arange(25, dtype=float).reshape(5, 5)
        glyph = ArrayGlyph(arr)
        glyph.plot(vmin=5.0, vmax=15.0)
        assert glyph.vmin == 5.0
        assert glyph.vmax == 15.0

    def test_invalid_kwarg_raises(self):
        """Plotting with an unknown kwarg raises `ValueError`."""
        glyph = ArrayGlyph(np.arange(9).reshape(3, 3))
        with pytest.raises(ValueError, match="not correct"):
            glyph.plot(banana_count=12)

    def test_recompute_with_explicit_ticks_spacing(self):
        """`plot(robust=True, ticks_spacing=...)` keeps the user's spacing."""
        rng = np.random.default_rng(1337)
        body = rng.random(98)
        arr = np.concatenate([body, [-1000.0, 1000.0]]).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        glyph.plot(robust=True, ticks_spacing=0.1)
        assert (
            glyph.default_options["ticks_spacing"] == 0.1
        ), "Explicit ticks_spacing must win over the recompute path"


@pytest.mark.plot
class TestPlotIdempotenceAndPurity:
    """Tests confirming repeated plot calls remain safe."""

    def test_replot_same_kind_does_not_raise(self):
        """Calling `plot(kind="contour")` twice produces equivalent state."""
        arr = np.arange(25, dtype=float).reshape(5, 5)
        glyph = ArrayGlyph(arr)
        fig1, ax1 = glyph.plot(kind="contour")
        # Reset fig/ax so the second call creates a fresh axes (the
        # method does not auto-create a new fig if one exists).
        glyph.fig = None
        glyph.ax = None
        fig2, ax2 = glyph.plot(kind="contour")
        assert isinstance(fig1, Figure)
        assert isinstance(fig2, Figure)
        assert len(ax1.collections) >= 1
        assert len(ax2.collections) >= 1


@pytest.mark.plot
class TestPlotRoundTripSavefig:
    """Round-trip test: rendered figure -> savefig -> non-empty PNG file."""

    def test_savefig_produces_nonempty_png(self, tmp_path):
        """Rendering and saving a figure yields a non-empty PNG file."""
        arr = np.arange(25, dtype=float).reshape(5, 5)
        glyph = ArrayGlyph(arr)
        fig, ax = glyph.plot()
        out = tmp_path / "array.png"
        try:
            fig.savefig(out)
            assert out.exists(), f"Expected {out} to exist after savefig"
            assert out.stat().st_size > 0, "PNG should be non-empty"
        finally:
            plt.close(fig)


@pytest.mark.plot
class TestAnimateEdgeCases:
    """Edge cases for :py`ArrayGlyph.animate`."""

    def test_invalid_kwarg_raises(
        self, coello_data: np.ndarray, animate_time_list: list
    ):
        """An unknown kwarg to `animate` raises `ValueError`."""
        glyph = ArrayGlyph(coello_data)
        with pytest.raises(ValueError, match="not correct"):
            glyph.animate(animate_time_list, banana=42)

    def test_explicit_vmin_vmax_in_animate(
        self, coello_data: np.ndarray, animate_time_list: list, no_data_value: float
    ):
        """`animate(vmin=, vmax=)` overrides constructor-derived limits."""
        glyph = ArrayGlyph(coello_data, exclude_value=[no_data_value])
        anim = glyph.animate(animate_time_list, vmin=10.0, vmax=200.0)
        assert anim is not None
        assert glyph.default_options["vmin"] == 10.0
        assert glyph.default_options["vmax"] == 200.0

    def test_animate_with_points(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        """`animate(points=...)` adds scatter overlays without raising."""
        glyph = ArrayGlyph(coello_data, exclude_value=[no_data_value])
        points = np.array([[1.0, 0, 0], [2.0, 1, 1]])
        anim = glyph.animate(animate_time_list, points=points)
        assert anim is not None

    def test_animate_with_explicit_ticks_spacing(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        """`animate(ticks_spacing=...)` overrides the auto-computed spacing."""
        glyph = ArrayGlyph(coello_data, exclude_value=[no_data_value])
        anim = glyph.animate(animate_time_list, ticks_spacing=50.0)
        assert anim is not None
        assert glyph.default_options["ticks_spacing"] == 50.0

    def test_animate_with_label_color(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        """`animate(label_color=...)` sets the frame label's text color."""
        glyph = ArrayGlyph(coello_data, exclude_value=[no_data_value])
        anim = glyph.animate(animate_time_list, label_color="yellow")
        assert anim is not None
        assert glyph._day_text.get_color() == "yellow"

    def test_label_color_is_keyword_only(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        """`label_color` must be keyword-only so no positional argument shifts."""
        glyph = ArrayGlyph(coello_data, exclude_value=[no_data_value])
        with pytest.raises(TypeError):
            glyph.animate(
                animate_time_list,
                None,
                ("white", "black"),
                200,
                None,
                "red",
                100,
                "blue",
                10,
                "yellow",
            )


@pytest.mark.plot
class TestNanNoDataConvention:
    """The NaN-as-nodata convention: a float raster (NetCDF float var, float
    GeoTIFF, …) that uses `np.nan` for no-data, loaded with the natural
    `ArrayGlyph(arr)` call (no `exclude_value` → defaults to `np.nan`).

    By design the NaN cells are *not* added to the mask (`array == np.nan` is
    `False` anyway, and matplotlib already renders NaN as blank), but
    `num_domain_cells` and the per-cell value artists must still agree on the
    count of display-eligible cells so `animate(display_cell_value=True)` does
    not blow up with an `IndexError` (regression for P1).
    """

    @staticmethod
    def _nan_raster(n: int = 4, h: int = 5, w: int = 5) -> np.ndarray:
        """Float stack of distinct values with two NaN no-data cells in
        frame 0 (the NaN-as-nodata convention)."""
        rng = np.random.default_rng(0)
        arr = rng.random((n, h, w)) * 100.0
        arr[0, 0, 0] = np.nan
        arr[0, 1, 1] = np.nan
        return arr

    def test_default_exclude_value_leaves_nan_unmasked(self):
        """The default `exclude_value` (`np.nan`) intentionally does not add
        the NaN cells to the mask — matplotlib renders NaN as blank, and the
        colour limits already go through `np.nanmin` / `np.nanmax`."""
        arr = self._nan_raster(n=1)
        glyph = ArrayGlyph(arr)
        mask = np.ma.getmaskarray(glyph.arr)
        assert not mask.any(), (
            f"NaN cells should be left unmasked by the default exclude_value; "
            f"mask has {int(mask.sum())} masked cell(s)"
        )

    def test_num_domain_cells_excludes_nan_cells(self):
        """`num_domain_cells` counts cells in the domain — neither masked nor
        NaN — so a 5x5 frame with 2 NaN no-data cells reports 23, matching the
        number of per-cell `Text` artists `plot` / `animate` create."""
        arr = self._nan_raster(n=1)  # 1x5x5, frame 0 has 2 NaN cells
        glyph = ArrayGlyph(arr)
        expected = arr[0].size - 2  # 23
        assert glyph.num_domain_cells == expected, (
            f"num_domain_cells should exclude the 2 NaN no-data cells "
            f"(expected {expected}), got {glyph.num_domain_cells}"
        )

    def test_animate_display_cell_value_true_with_nan_nodata(self):
        """`animate(display_cell_value=True)` on a NaN-nodata stack runs
        without `IndexError` (the cell-update loop iterates the artist list,
        not `num_domain_cells`). Regression for P1."""
        arr = self._nan_raster(n=4)
        glyph = ArrayGlyph(arr)
        anim = glyph.animate(time=list(range(4)), display_cell_value=True)
        assert isinstance(anim, FuncAnimation)
        plt.close(glyph.fig)

    def test_plot_display_cell_value_true_with_nan_nodata(self):
        """`plot(display_cell_value=True)` on the same raster renders one
        `Text` artist per non-NaN cell and does not crash."""
        arr = self._nan_raster(n=1)[0]  # 2-D 5x5 frame
        glyph = ArrayGlyph(arr)
        fig, ax = glyph.plot(display_cell_value=True)
        try:
            assert isinstance(fig, Figure)
            n_nan = int(np.isnan(arr).sum())
            # ax.texts holds the per-cell value annotations (title / axis
            # labels live elsewhere); one per non-NaN cell.
            assert len(ax.texts) == arr.size - n_nan, (
                f"expected {arr.size - n_nan} cell-value texts (NaN cells "
                f"skipped), got {len(ax.texts)}"
            )
        finally:
            plt.close(fig)


class TestExcludeValueMultiValue:
    """Cover the `len(exclude_value) > 1` constructor branch."""

    def test_two_exclude_values_masks_both(self):
        """Two exclude values should mask both target cells via `logical_or`."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        glyph = ArrayGlyph(arr, exclude_value=[1.0, 9.0])
        assert glyph.arr.mask[0, 0], "Cell with value 1 must be masked"
        assert glyph.arr.mask[2, 2], "Cell with value 9 must be masked"
        assert not glyph.arr.mask[1, 1], "Middle cell must remain unmasked"


class TestPrepareArrayPercentilePath:
    """Cover the `percentile` branch in `prepare_array`."""

    def test_percentile_drives_scaling(self):
        """`percentile=2` triggers the `scale_percentile` branch."""
        rng = np.random.default_rng(1337)
        arr = rng.integers(0, 10000, size=(3, 4, 4)).astype(np.float32)
        glyph = ArrayGlyph(np.zeros((1, 1)))
        result = glyph.prepare_array(arr, rgb=[0, 1, 2], percentile=2)
        assert result.shape == (4, 4, 3)
        assert np.all((0.0 <= result) & (result <= 1.0))


@pytest.mark.plot
class TestContourLevelsAndNorm:
    """Cover contour/contourf branches with norm and levels."""

    def test_contour_with_norm_and_levels(self):
        """`contour` with a non-None norm and integer levels uses both paths."""
        arr = np.linspace(0.0, 1.0, 100).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        fig, ax = glyph.plot(
            kind="contour",
            color_scale="power",
            levels=4,
        )
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1

    def test_contourf_with_explicit_levels(self):
        """`contourf` with explicit edges uses the `level_edges` branch."""
        arr = np.linspace(0.0, 1.0, 100).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        fig, ax = glyph.plot(
            kind="contourf",
            levels=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1

    def test_contour_with_masked_array(self):
        """A masked-array input is filled with NaN before contour rendering."""
        arr = np.linspace(0.0, 1.0, 100).reshape(10, 10)
        # Use exclude_value path so the constructor builds a masked array.
        arr[0, 0] = -999.0
        glyph = ArrayGlyph(arr, exclude_value=[-999.0])
        fig, ax = glyph.plot(kind="contour")
        assert isinstance(fig, Figure)

    def test_contour_with_midpoint_already_filled(self):
        """`color_scale='midpoint'` pre-fills the array; contour skips the mask branch."""
        arr = np.linspace(-1.0, 1.0, 100).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        fig, ax = glyph.plot(kind="contour", color_scale="midpoint", midpoint=0.0)
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1


@pytest.mark.plot
class TestContourLabels:
    """`plot(kind="contour", labels=...)` inline label support (issue #148)."""

    @staticmethod
    def _smooth_arr() -> np.ndarray:
        """A smooth 2-D field that yields several labelled isolines."""
        y, x = np.mgrid[-3:3:30j, -3:3:30j]
        return np.exp(-(x**2 + y**2))

    def test_labels_true_draws_and_exposes_text(self):
        """`labels=True` populates `contour_labels` with `Text` artists."""
        glyph = ArrayGlyph(self._smooth_arr())
        fig, ax = glyph.plot(kind="contour", labels=True)
        assert isinstance(fig, Figure)
        assert isinstance(glyph.contour_labels, list)
        assert len(glyph.contour_labels) > 0
        assert all(isinstance(t, Text) for t in glyph.contour_labels)
        # The label artists are attached to the axes' text list.
        assert set(glyph.contour_labels).issubset(set(ax.texts))

    def test_labels_default_is_noop(self):
        """Default (`labels=False`) draws no labels; `contour_labels` is None."""
        glyph = ArrayGlyph(self._smooth_arr())
        fig, ax = glyph.plot(kind="contour")
        assert glyph.contour_labels is None
        assert len(ax.texts) == 0

    def test_label_kw_forwarded_to_clabel(self):
        """`label_kw` reaches `ax.clabel` (custom `fmt`/`fontsize` applied)."""
        glyph = ArrayGlyph(self._smooth_arr())
        fig, ax = glyph.plot(
            kind="contour",
            labels=True,
            label_kw={"fmt": "%.3f", "fontsize": 6},
        )
        assert len(glyph.contour_labels) > 0
        # The custom fontsize on every label proves label_kw was forwarded.
        assert all(t.get_fontsize() == 6 for t in glyph.contour_labels)
        # The "%.3f" format yields three decimal places in every label.
        assert all(re.search(r"\.\d{3}$", t.get_text()) for t in glyph.contour_labels)

    def test_labels_true_on_contourf_is_noop(self):
        """`labels=True` is ignored for `contourf` (no isolines to label)."""
        glyph = ArrayGlyph(self._smooth_arr())
        fig, ax = glyph.plot(kind="contourf", labels=True)
        assert isinstance(fig, Figure)
        assert glyph.contour_labels is None

    def test_replot_without_labels_resets_contour_labels(self):
        """Re-plotting with `labels=False` clears a prior render's labels."""
        glyph = ArrayGlyph(self._smooth_arr())
        glyph.plot(kind="contour", labels=True)
        assert glyph.contour_labels is not None
        glyph.plot(kind="contour", labels=False)
        assert glyph.contour_labels is None

    def test_switching_kind_resets_contour_labels(self):
        """A subsequent non-contour render clears stale label artists."""
        glyph = ArrayGlyph(self._smooth_arr())
        glyph.plot(kind="contour", labels=True)
        assert glyph.contour_labels is not None
        glyph.plot(kind="imshow")
        assert glyph.contour_labels is None

    def test_labels_on_constant_field_is_empty_list(self):
        """A labelled contour with no isolines yields [] (not None), no raise."""
        glyph = ArrayGlyph(np.full((5, 5), 3.0))
        with warnings.catch_warnings():
            # A constant field has no contour lines; the colorbar-skip
            # warning is expected and unrelated to labelling.
            warnings.simplefilter("ignore")
            fig, ax = glyph.plot(kind="contour", labels=True)
        assert isinstance(fig, Figure)
        assert glyph.contour_labels == []

    def test_label_kw_overrides_cleopatra_defaults(self):
        """User `label_kw` keys win over cleopatra's clabel defaults."""
        glyph = ArrayGlyph(self._smooth_arr())
        # Default fontsize is 8; the user value must take precedence.
        fig, ax = glyph.plot(kind="contour", labels=True, label_kw={"fontsize": 14})
        assert len(glyph.contour_labels) > 0
        assert all(t.get_fontsize() == 14 for t in glyph.contour_labels)

    def test_contour_labels_none_before_any_render(self):
        """`contour_labels` is None on a freshly constructed glyph.

        Test scenario:
            The attribute is initialised in `__init__`, so it must be
            reachable and `None` before `plot`/`animate` is ever called.
        """
        glyph = ArrayGlyph(self._smooth_arr())
        assert (
            glyph.contour_labels is None
        ), f"Expected None before render, got {glyph.contour_labels!r}"

    @pytest.mark.parametrize("kind", ["imshow", "pcolormesh", "auto"])
    def test_labels_true_is_noop_for_non_contour_kinds(self, kind):
        """`labels=True` is silently ignored for every non-contour kind.

        Args:
            kind: A render kind that draws no isolines.

        Test scenario:
            `imshow`/`pcolormesh`/`auto` (which routes to `imshow` here)
            must not draw labels; `contour_labels` stays None. Complements
            the dedicated `contourf` no-op test.
        """
        glyph = ArrayGlyph(self._smooth_arr())
        fig, ax = glyph.plot(kind=kind, labels=True)
        assert isinstance(fig, Figure)
        assert (
            glyph.contour_labels is None
        ), f"kind={kind!r} should not draw labels, got {glyph.contour_labels!r}"

    def test_label_kw_forwards_arbitrary_clabel_kwarg(self):
        """A non-default `label_kw` key (`colors`) reaches `ax.clabel`.

        Test scenario:
            Proves the passthrough is not limited to `fmt`/`fontsize`: a
            `colors="red"` entry must colour every label red.
        """
        glyph = ArrayGlyph(self._smooth_arr())
        fig, ax = glyph.plot(kind="contour", labels=True, label_kw={"colors": "red"})
        assert len(glyph.contour_labels) > 0
        red = to_rgba("red")
        assert all(
            to_rgba(t.get_color()) == red for t in glyph.contour_labels
        ), "every label should be red when label_kw={'colors': 'red'}"

    def test_labels_with_curvilinear_coords(self):
        """`labels=True` works on the `coords=` (curvilinear) contour path.

        Test scenario:
            With `coords` supplied the contour branch forwards `(x, y, z)`
            positionally; labelling must still populate `contour_labels`.
        """
        arr = self._smooth_arr()
        ny, nx = arr.shape
        x = np.linspace(-3.0, 3.0, nx)
        y = np.linspace(-3.0, 3.0, ny)
        glyph = ArrayGlyph(arr, coords=(x, y))
        fig, ax = glyph.plot(kind="contour", labels=True)
        assert len(glyph.contour_labels) > 0
        assert all(isinstance(t, Text) for t in glyph.contour_labels)

    def test_labels_with_discrete_levels(self):
        """`labels=True` works on the `levels=` (level-edges) contour path.

        Test scenario:
            Setting integer `levels` routes through the `level_edges`
            branch; labels must still be drawn and exposed.
        """
        glyph = ArrayGlyph(self._smooth_arr())
        fig, ax = glyph.plot(kind="contour", labels=True, levels=5)
        assert len(glyph.contour_labels) > 0
        assert all(isinstance(t, Text) for t in glyph.contour_labels)

    def test_labels_with_masked_array(self):
        """`labels=True` works when the source is a masked array.

        Test scenario:
            An `exclude_value` builds a masked array that the contour path
            fills with NaN before drawing; labelling must still succeed.
        """
        arr = self._smooth_arr().copy()
        arr[0, 0] = -999.0
        glyph = ArrayGlyph(arr, exclude_value=[-999.0])
        fig, ax = glyph.plot(kind="contour", labels=True)
        assert len(glyph.contour_labels) > 0
        assert all(isinstance(t, Text) for t in glyph.contour_labels)


@pytest.mark.plot
class TestCenterCmapDoesNotApplyToRgb:
    """RGB compositing path skips the center/cmap auto-switch logic."""

    def test_rgb_arr_renders_without_norm(self):
        """An RGB array renders via `imshow` and ignores center logic."""
        rgb_arr = (
            np.random.default_rng(0).integers(0, 255, size=(3, 8, 8)).astype(np.float32)
        )
        glyph = ArrayGlyph(rgb_arr, rgb=[0, 1, 2])
        fig, ax = glyph.plot()
        assert isinstance(fig, Figure)
        # No colorbar is created on the RGB path; `cbar` stays None.
        assert glyph.cbar is None, "RGB compositing should not create a colorbar"


@pytest.mark.plot
class TestCoordsCurvilinear:
    """Tests for curvilinear / non-uniform `coords=(x, y)` support (CLEO-2).

    Covers the `coords` constructor kwarg introduced in C-2, the
    automatic `kind="auto"` -> `pcolormesh` dispatch when coords
    are present, mutual exclusivity with `extent`, and shape
    validation.
    """

    @staticmethod
    def _arr_3x4() -> np.ndarray:
        """Small `(3 rows, 4 cols)` test array."""
        return np.arange(12, dtype=float).reshape(3, 4)

    def test_1d_coords_auto_routes_to_pcolormesh(self):
        """1-D `coords` + `kind="auto"` renders as a `QuadMesh`."""
        arr = self._arr_3x4()
        x = np.linspace(0.0, 10.0, 4)
        y = np.linspace(0.0, 5.0, 3)
        glyph = ArrayGlyph(arr, coords=(x, y))
        fig, ax = glyph.plot(kind="auto")
        assert isinstance(fig, Figure)
        # `pcolormesh` produces a QuadMesh that lives in ax.collections.
        assert len(ax.collections) >= 1
        # And not an image (imshow would have been the wrong path).
        assert len(ax.get_images()) == 0

    def test_2d_meshgrid_coords_render_via_pcolormesh(self):
        """2-D `(X, Y)` from `np.meshgrid` renders via `pcolormesh`."""
        arr = self._arr_3x4()
        x1d = np.linspace(0.0, 10.0, 4)
        y1d = np.linspace(0.0, 5.0, 3)
        x2d, y2d = np.meshgrid(x1d, y1d)
        glyph = ArrayGlyph(arr, coords=(x2d, y2d))
        fig, ax = glyph.plot(kind="pcolormesh")
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1

    def test_imshow_with_coords_raises(self):
        """`kind="imshow"` together with `coords` raises `ValueError`.

        Document the design choice: imshow renders on the array's
        index grid and cannot honour arbitrary (x, y) coordinate
        arrays. Callers must use `kind="pcolormesh"` or
        `kind="auto"` instead.
        """
        arr = self._arr_3x4()
        x = np.linspace(0.0, 10.0, 4)
        y = np.linspace(0.0, 5.0, 3)
        glyph = ArrayGlyph(arr, coords=(x, y))
        with pytest.raises(ValueError, match="coords"):
            glyph.plot(kind="imshow")

    def test_coord_shape_mismatch_raises(self):
        """A coord array whose shape does not match the data raises."""
        arr = self._arr_3x4()
        x_bad = np.linspace(0.0, 10.0, 99)  # length mismatch
        y = np.linspace(0.0, 5.0, 3)
        with pytest.raises(ValueError, match=re.escape(_COORD_SHAPE_MISMATCH)):
            ArrayGlyph(arr, coords=(x_bad, y))

    def test_extent_and_coords_together_raises(self):
        """Passing both `extent` and `coords` raises `ValueError`."""
        arr = self._arr_3x4()
        x = np.linspace(0.0, 10.0, 4)
        y = np.linspace(0.0, 5.0, 3)
        with pytest.raises(ValueError, match="mutually exclusive"):
            ArrayGlyph(
                arr,
                extent=[0.0, 0.0, 10.0, 5.0],
                coords=(x, y),
            )

    def test_coords_not_two_tuple_raises(self):
        """`coords` must be a length-2 sequence; otherwise `TypeError`."""
        arr = self._arr_3x4()
        with pytest.raises(TypeError, match="length-2 sequence"):
            ArrayGlyph(arr, coords="oops")

    def test_2d_coords_with_contourf(self):
        """Curvilinear 2-D coords also forward through `kind="contourf"`."""
        arr = self._arr_3x4()
        x1d = np.linspace(0.0, 10.0, 4)
        y1d = np.linspace(0.0, 5.0, 3)
        x2d, y2d = np.meshgrid(x1d, y1d)
        glyph = ArrayGlyph(arr, coords=(x2d, y2d))
        fig, ax = glyph.plot(kind="contourf")
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1

    def test_backward_compat_no_coords_renders_imshow(self):
        """An `ArrayGlyph(arr)` with no coords still renders via imshow."""
        arr = self._arr_3x4()
        glyph = ArrayGlyph(arr)
        assert glyph.coords is None
        fig, ax = glyph.plot()
        assert isinstance(fig, Figure)
        # No coords -> auto resolves to imshow, which produces an
        # AxesImage rather than a QuadMesh.
        assert len(ax.get_images()) >= 1


@pytest.mark.plot
class TestFaceting:
    """Tests for the faceting API introduced in CLEO-4 (`ArrayGlyph.facet`)."""

    @staticmethod
    def _stack_3d(n: int = 4, h: int = 10, w: int = 10) -> np.ndarray:
        """Build a 3-D `(n, h, w)` stack of distinct frames."""
        rng = np.random.default_rng(seed=42)
        return rng.uniform(0.0, 1.0, size=(n, h, w))

    @staticmethod
    def _stack_4d(
        n_col: int = 2, n_row: int = 3, h: int = 10, w: int = 10
    ) -> np.ndarray:
        """Build a 4-D `(n_col, n_row, h, w)` stack."""
        rng = np.random.default_rng(seed=43)
        return rng.uniform(0.0, 1.0, size=(n_col, n_row, h, w))

    def test_3d_col_only_builds_single_row(self):
        """3-D stack with `col="t"` yields a 1xN grid of subplots."""
        stack = self._stack_3d(n=4)
        result = ArrayGlyph(stack).facet(col="t")
        assert isinstance(result, FacetGrid)
        assert result.axes.shape == (1, 4)
        # Each subplot has rendered an artist (imshow -> images).
        for ax in result.axes.ravel():
            assert len(ax.get_images()) >= 1

    def test_col_wrap_produces_wrapped_grid(self):
        """`col_wrap=3` on 6 panels yields a 2x3 grid."""
        stack = self._stack_3d(n=6)
        result = ArrayGlyph(stack).facet(col="t", col_wrap=3)
        assert result.axes.shape == (2, 3)
        # All 6 panels visible.
        visible = [ax for ax in result.axes.ravel() if ax.get_visible()]
        assert len(visible) == 6

    def test_4d_col_row_grid(self):
        """4-D stack with `col` + `row` yields a NrowxNcol grid."""
        stack = self._stack_4d(n_col=2, n_row=3)
        result = ArrayGlyph(stack).facet(col="t", row="level")
        # nrows=3 (from n_row), ncols=2 (from n_col).
        assert result.axes.shape == (3, 2)
        assert len(result.name_dicts) == 6

    def test_shared_vmin_vmax_across_subplots(self):
        """Every subplot must share the same `norm.vmin`/`norm.vmax`."""
        stack = self._stack_3d(n=4)
        # Pin explicit limits so the assertion is robust to the
        # internal stack-wide computation.
        result = ArrayGlyph(stack).facet(col="t", vmin=0.0, vmax=1.0)
        first = result.axes.ravel()[0].get_images()[0]
        for ax in result.axes.ravel():
            ims = ax.get_images()
            assert len(ims) >= 1
            assert ims[0].norm.vmin == first.norm.vmin
            assert ims[0].norm.vmax == first.norm.vmax
        assert first.norm.vmin == 0.0
        assert first.norm.vmax == 1.0

    def test_shared_colorbar_attached(self):
        """The returned `FacetGrid` exposes a single shared colorbar."""
        stack = self._stack_3d(n=3)
        result = ArrayGlyph(stack).facet(col="t")
        # `cbar` is taken from the first rendered subplot; not None
        # for the non-RGB path.
        assert result.cbar is not None

    def test_coord_aware_titles(self):
        """`col_coords` plugs into the per-subplot title and name_dicts."""
        stack = self._stack_3d(n=4)
        coords = [0, 6, 12, 18]
        result = ArrayGlyph(stack).facet(col="hour", col_coords=coords)
        # Each title must reference the coord value, not the index.
        for ax, want in zip(result.axes.ravel(), coords):
            assert str(want) in ax.get_title()
        # And `name_dicts` mirrors xarray's structure.
        assert result.name_dicts[0] == {"hour": 0}
        assert result.name_dicts[-1] == {"hour": 18}

    def test_no_col_no_row_raises(self):
        """Calling `facet()` without `col` or `row` raises `ValueError`."""
        stack = self._stack_3d(n=3)
        with pytest.raises(ValueError, match="at least one of"):
            ArrayGlyph(stack).facet()

    def test_savefig_roundtrip(self, tmp_path):
        """Rendering and saving a facet figure yields a non-empty PNG."""
        stack = self._stack_3d(n=4)
        result = ArrayGlyph(stack).facet(col="t")
        out = tmp_path / "facet.png"
        try:
            result.fig.savefig(out)
            assert out.exists()
            assert out.stat().st_size > 0
        finally:
            plt.close(result.fig)

    def test_col_wrap_hides_trailing_empty_slots(self):
        """When `col_wrap` does not divide N, trailing axes are hidden."""
        stack = self._stack_3d(n=5)
        result = ArrayGlyph(stack).facet(col="t", col_wrap=3)
        # Layout: 3 cols x 2 rows = 6 slots, 5 panels rendered.
        assert result.axes.shape == (2, 3)
        hidden = [ax for ax in result.axes.ravel() if not ax.get_visible()]
        assert len(hidden) == 1

    def test_masked_stack_preserves_mask_per_panel(self):
        """Masked-array stacks keep their per-panel mask in `facet`.

        Regression for H1: `np.asarray(arr[col_idx])` previously
        dropped the mask, so cells excluded via `exclude_value` were
        rendered as real data on every subplot.
        """
        import numpy.ma as ma

        stack = self._stack_3d(n=3)
        sentinel = -9999.0
        stack[0, 0, 0] = sentinel
        stack[1, 1, 1] = sentinel
        stack[2, 2, 2] = sentinel

        result = ArrayGlyph(stack, exclude_value=[sentinel]).facet(col="t")
        try:
            for panel_idx, ax in enumerate(result.axes.ravel()):
                images = ax.get_images()
                assert images, f"panel {panel_idx} must have an AxesImage"
                arr_on_ax = images[0].get_array()
                assert isinstance(arr_on_ax, ma.MaskedArray), (
                    f"panel {panel_idx} lost its MaskedArray wrapper; "
                    f"got {type(arr_on_ax).__name__}"
                )
                mask = ma.getmaskarray(arr_on_ax)
                assert mask[panel_idx, panel_idx], (
                    f"panel {panel_idx} should keep sentinel cell "
                    f"({panel_idx}, {panel_idx}) masked; mask={mask}"
                )
        finally:
            plt.close(result.fig)

    def test_preserves_dtype_per_panel(self):
        """Per-panel slices keep the stack's dtype — no `np.asarray` upcast (M6)."""
        stack = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)
        result = ArrayGlyph(stack).facet(col="t")
        try:
            for panel_idx, ax in enumerate(result.axes.flat):
                arr_on_ax = ax.get_images()[0].get_array()
                assert arr_on_ax.dtype == np.uint8, (
                    f"panel {panel_idx} dtype changed to {arr_on_ax.dtype}, "
                    "expected uint8"
                )
        finally:
            plt.close(result.fig)


@pytest.mark.plot
class TestAnimateCbarAttribute:
    """`animate` exposes the colorbar on the instance, like `plot` (M5)."""

    def test_animate_sets_cbar(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        """After `animate(...)` the glyph has a `.cbar` (a matplotlib Colorbar)."""
        from matplotlib.colorbar import Colorbar

        glyph = ArrayGlyph(coello_data, exclude_value=[no_data_value])
        anim = glyph.animate(animate_time_list)
        try:
            assert hasattr(glyph, "cbar"), "animate should set glyph.cbar"
            assert isinstance(glyph.cbar, Colorbar)
        finally:
            plt.close(anim._fig)


@pytest.mark.plot
class TestFacetExtents:
    """Per-panel `extents=` on :py`ArrayGlyph.facet` (M4)."""

    @staticmethod
    def _stack(n: int = 2, h: int = 4, w: int = 4) -> np.ndarray:
        return np.arange(n * h * w, dtype=float).reshape(n, h, w)

    def test_per_panel_extents_applied(self):
        """Each panel's image gets its own `[xmin, ymin, xmax, ymax]`."""
        stack = self._stack(n=2)
        result = ArrayGlyph(stack).facet(
            col="region", extents=[[0, 0, 10, 10], [10, 0, 20, 10]]
        )
        try:
            extents = [
                tuple(ax.get_images()[0].get_extent()) for ax in result.axes.flat
            ]
            # matplotlib reports extent as (xmin, xmax, ymin, ymax).
            assert extents == [(0.0, 10.0, 0.0, 10.0), (10.0, 20.0, 0.0, 10.0)]
        finally:
            plt.close(result.fig)

    def test_per_panel_extents_4d(self):
        """`extents` covers all panels of a 4-D (col x row) grid, row-major."""
        stack = np.arange(2 * 2 * 3 * 3, dtype=float).reshape(2, 2, 3, 3)
        ex = [[0, 0, 1, 1], [1, 0, 2, 1], [0, 1, 1, 2], [1, 1, 2, 2]]
        result = ArrayGlyph(stack).facet(col="c", row="r", extents=ex)
        try:
            got = [tuple(ax.get_images()[0].get_extent()) for ax in result.axes.flat]
            assert got == [
                (0.0, 1.0, 0.0, 1.0),
                (1.0, 2.0, 0.0, 1.0),
                (0.0, 1.0, 1.0, 2.0),
                (1.0, 2.0, 1.0, 2.0),
            ]
        finally:
            plt.close(result.fig)

    def test_extents_wrong_length_raises(self):
        """An `extents` list whose length != n_panels raises `ValueError`."""
        stack = self._stack(n=3)
        with pytest.raises(ValueError, match="3 panels"):
            ArrayGlyph(stack).facet(col="t", extents=[[0, 0, 1, 1]])

    def test_extents_non_length4_element_raises(self):
        """An `extents` entry that isn't length-4 raises `ValueError`."""
        stack = self._stack(n=2)
        with pytest.raises(ValueError, match=r"extents\[1\].*length-4"):
            ArrayGlyph(stack).facet(col="t", extents=[[0, 0, 1, 1], [0, 0, 1]])

    def test_extents_with_parent_extent_raises(self):
        """`extents` and the glyph's own `extent` are mutually exclusive."""
        stack = self._stack(n=2)
        with pytest.raises(ValueError, match="mutually exclusive"):
            ArrayGlyph(stack, extent=[0, 0, 4, 4]).facet(
                col="t", extents=[[0, 0, 1, 1], [1, 0, 2, 1]]
            )

    def test_extents_with_coords_raises(self):
        """`extents` and `coords` are mutually exclusive."""
        stack = self._stack(n=2, h=3, w=4)
        x = np.linspace(0.0, 10.0, 4)
        y = np.linspace(0.0, 5.0, 3)
        with pytest.raises(ValueError, match="mutually exclusive"):
            ArrayGlyph(stack, coords=(x, y)).facet(
                col="t", extents=[[0, 0, 1, 1], [1, 0, 2, 1]]
            )

    def test_no_extents_reuses_parent_extent(self):
        """Without `extents` every panel inherits the parent's `extent`."""
        stack = self._stack(n=3)
        result = ArrayGlyph(stack, extent=[0, 0, 4, 8]).facet(col="t")
        try:
            for ax in result.axes.flat:
                # parent extent [xmin, ymin, xmax, ymax] -> matplotlib
                # (xmin, xmax, ymin, ymax)
                assert tuple(ax.get_images()[0].get_extent()) == (0.0, 4.0, 0.0, 8.0)
        finally:
            plt.close(result.fig)


@pytest.mark.plot
class TestAnimateDataGetter:
    """Tests for the `data_getter` callback added to `animate` (CLEO-7)."""

    @staticmethod
    def _stack(n: int = 4, h: int = 5, w: int = 5) -> np.ndarray:
        rng = np.random.default_rng(seed=7)
        return rng.uniform(0.0, 1.0, size=(n, h, w))

    def test_data_getter_supplies_frames(self):
        """With `data_getter` set, the callback drives the animation."""
        stack = self._stack(n=4)
        # 2-D template — the animate path uses it only for shape.
        template = stack[0]
        glyph = ArrayGlyph(template)
        anim = glyph.animate(
            time=list(range(4)),
            data_getter=lambda i: stack[i],
        )
        assert isinstance(anim, FuncAnimation)

    def test_data_getter_wrong_shape_raises(self):
        """A callback that returns the wrong shape raises `ValueError`."""
        stack = self._stack(n=4)
        template = stack[0]
        glyph = ArrayGlyph(template)
        with pytest.raises(ValueError, match="do not match"):
            glyph.animate(
                time=list(range(4)),
                data_getter=lambda i: np.zeros((99, 99)),
            )

    def test_data_getter_none_falls_back_to_self_arr(
        self,
        coello_data: np.ndarray,
        animate_time_list: list,
        no_data_value: float,
    ):
        """`data_getter=None` preserves the existing 3-D arr path."""
        glyph = ArrayGlyph(coello_data, exclude_value=[no_data_value])
        anim = glyph.animate(animate_time_list, data_getter=None)
        assert isinstance(anim, FuncAnimation)

    def test_2d_arr_without_data_getter_raises(self):
        """A 2-D `self.arr` without a callback raises `ValueError`."""
        glyph = ArrayGlyph(np.zeros((5, 5)))
        with pytest.raises(ValueError, match="data_getter callback"):
            glyph.animate(time=list(range(3)))

    def test_data_getter_invocation_count(self, tmp_path):
        """`data_getter` is invoked once per rendered frame.

        Save the animation to a temp GIF to force matplotlib to walk
        every frame; the closure counter must reach `len(time)`.
        """
        stack = self._stack(n=4)
        template = stack[0]
        glyph = ArrayGlyph(template)
        counter = {"n": 0}

        def getter(i):
            counter["n"] += 1
            return stack[i]

        anim = glyph.animate(
            time=list(range(4)),
            data_getter=getter,
        )
        out = tmp_path / "anim.gif"
        try:
            glyph.save_animation(str(out), fps=2)
            assert counter["n"] >= 4, (
                f"`data_getter` should be invoked at least once per frame; "
                f"got {counter['n']} calls for 4 frames"
            )
        finally:
            plt.close(anim._fig)


@pytest.mark.plot
class TestCoordsCurvilinearEdgeCases:
    """Edge-case coverage for the curvilinear `coords=(x, y)` path (Phase 2 / CLEO-2).

    Complements `TestCoordsCurvilinear` with NaN handling, descending
    coordinates, dtype variants, small/degenerate grids, and an explicit
    `kind="pcolormesh"` + `coords=None` round-trip.
    """

    @staticmethod
    def _arr_3x4() -> np.ndarray:
        """Build a small `(3, 4)` array suitable for curvilinear tests."""
        return np.arange(12, dtype=float).reshape(3, 4)

    def test_1d_coords_with_nan_rejected_by_matplotlib(self) -> None:
        """1-D coords with NaN cell centres are rejected at render time.

        Test scenario:
            `_validate_coords` happily stores NaN-containing arrays
            (it only checks shapes), but matplotlib's `pcolormesh`
            raises on non-finite coordinates. This documents that
            contract.
        """
        arr = self._arr_3x4()
        x = np.array([0.0, 1.0, np.nan, 3.0])
        y = np.array([0.0, 1.0, 2.0])
        glyph = ArrayGlyph(arr, coords=(x, y))
        with pytest.raises(ValueError, match="non-finite"):
            glyph.plot(kind="auto")

    def test_descending_1d_coords_render(self) -> None:
        """Descending coordinates (e.g. lat N->S) are accepted and rendered.

        Test scenario:
            Latitude often runs north-to-south in NetCDF data;
            `pcolormesh` handles descending centres without raising.
        """
        arr = self._arr_3x4()
        x = np.linspace(10.0, 0.0, 4)
        y = np.linspace(5.0, 0.0, 3)
        glyph = ArrayGlyph(arr, coords=(x, y))
        fig, ax = glyph.plot(kind="auto")
        try:
            assert isinstance(
                fig, Figure
            ), "descending coords must still produce a Figure"
            assert len(ax.collections) >= 1, "expected at least one mesh artist"
        finally:
            plt.close(fig)

    def test_1d_coords_corner_length_rejected(self) -> None:
        """Corner-length 1-D coords (`shape[-1]+1`) are rejected.

        Test scenario:
            `_validate_coords` requires 1-D coord length to match the
            data axis (cell-centre semantics); passing the +1 corner
            count raises `ValueError`.
        """
        arr = self._arr_3x4()
        x_corner = np.linspace(0.0, 10.0, 5)
        y_corner = np.linspace(0.0, 5.0, 4)
        with pytest.raises(ValueError, match=re.escape(_COORD_SHAPE_MISMATCH)):
            ArrayGlyph(arr, coords=(x_corner, y_corner))

    def test_minimal_2x2_grid_renders(self) -> None:
        """A degenerate `2x2` grid with 1-D coords renders cleanly.

        Test scenario:
            The smallest meaningful pcolormesh case is a 2x2 grid; the
            curvilinear path must not assume a minimum size.
        """
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        glyph = ArrayGlyph(arr, coords=(x, y))
        fig, ax = glyph.plot(kind="auto")
        try:
            assert isinstance(fig, Figure), "2x2 grid must render"
            assert len(ax.collections) >= 1, "expected mesh on 2x2 grid"
        finally:
            plt.close(fig)

    def test_single_row_array_renders(self) -> None:
        """A `(1, N)` single-row array renders via pcolormesh.

        Test scenario:
            Degenerate single-row inputs occur for transect-style data;
            the curvilinear path must accept them.
        """
        arr = np.arange(4, dtype=float).reshape(1, 4)
        x = np.linspace(0.0, 10.0, 4)
        y = np.array([0.0])
        glyph = ArrayGlyph(arr, coords=(x, y))
        fig, ax = glyph.plot(kind="auto")
        try:
            assert isinstance(fig, Figure), "single-row array must render"
            assert len(ax.collections) >= 1, "expected pcolormesh on single row"
        finally:
            plt.close(fig)

    def test_single_column_array_renders(self) -> None:
        """A `(N, 1)` single-column array renders via pcolormesh.

        Test scenario:
            Single-column profile data (e.g. depth profile) is a valid
            curvilinear input.
        """
        arr = np.arange(3, dtype=float).reshape(3, 1)
        x = np.array([0.0])
        y = np.linspace(0.0, 5.0, 3)
        glyph = ArrayGlyph(arr, coords=(x, y))
        fig, ax = glyph.plot(kind="auto")
        try:
            assert isinstance(fig, Figure), "single-column array must render"
            assert len(ax.collections) >= 1, "expected pcolormesh on single column"
        finally:
            plt.close(fig)

    def test_pcolormesh_explicit_without_coords(self) -> None:
        """`kind="pcolormesh"` with `coords=None` falls back to array indices.

        Test scenario:
            Choosing pcolormesh explicitly while leaving coords unset
            must render on the array's index grid (no coord forwarding).
        """
        arr = self._arr_3x4()
        glyph = ArrayGlyph(arr)
        assert glyph.coords is None, "coords must default to None"
        fig, ax = glyph.plot(kind="pcolormesh")
        try:
            assert isinstance(fig, Figure), "explicit pcolormesh must render"
            assert len(ax.collections) >= 1, "expected QuadMesh artist"
        finally:
            plt.close(fig)

    @pytest.mark.parametrize(
        "dtype",
        [np.int32, np.int64, np.float32, np.float64],
        ids=["int32", "int64", "float32", "float64"],
    )
    def test_coords_dtype_variants(self, dtype) -> None:
        """Integer and float coord dtypes both render via pcolormesh.

        Args:
            dtype: numpy dtype applied to the coordinate arrays.

        Test scenario:
            Coordinate arrays may arrive as integer indices or 32/64-bit
            floats from upstream readers; the curvilinear path must
            not narrow the accepted dtype.
        """
        arr = self._arr_3x4()
        x = np.array([0, 1, 2, 3], dtype=dtype)
        y = np.array([0, 1, 2], dtype=dtype)
        glyph = ArrayGlyph(arr, coords=(x, y))
        fig, ax = glyph.plot(kind="auto")
        try:
            assert isinstance(fig, Figure), f"dtype {dtype} must render"
            assert glyph.coords[0].dtype == np.dtype(
                dtype
            ), f"coord dtype must be preserved; got {glyph.coords[0].dtype}"
        finally:
            plt.close(fig)

    def test_coords_unmasked_masked_array(self) -> None:
        """Unmasked masked-array coords flow through `np.asarray` and render.

        Test scenario:
            Coordinates from netCDF readers are sometimes wrapped in
            `numpy.ma.MaskedArray` with no actual mask. `np.asarray`
            in `_validate_coords` keeps the underlying values intact
            and pcolormesh accepts them.
        """
        arr = self._arr_3x4()
        x_raw = np.array([0.0, 1.0, 2.0, 3.0])
        x_ma = np.ma.array(x_raw, mask=False)
        y = np.linspace(0.0, 2.0, 3)
        glyph = ArrayGlyph(arr, coords=(x_ma, y))
        fig, ax = glyph.plot(kind="auto")
        try:
            assert isinstance(fig, Figure), "unmasked MaskedArray must render"
            assert len(ax.collections) >= 1, "expected mesh artist"
        finally:
            plt.close(fig)

    @pytest.mark.parametrize(
        "bad_x",
        [
            np.array([True, False, True, False]),
            np.array([0, 1, 2, 3], dtype=complex),
            np.array(["a", "b", "c", "d"], dtype=object),
        ],
        ids=["bool", "complex", "object"],
    )
    def test_non_numeric_coords_rejected(self, bad_x) -> None:
        """Non-numeric coord dtypes (bool / complex / object) raise `ValueError` (L6).

        Args:
            bad_x: An x-coord array with a dtype that should be rejected.
        """
        arr = self._arr_3x4()
        y = np.linspace(0.0, 2.0, 3)
        with pytest.raises(ValueError, match=re.escape(_COORD_DTYPE_MISMATCH)):
            ArrayGlyph(arr, coords=(bad_x, y))

    def test_coords_contourf_roundtrip_savefig(self, tmp_path) -> None:
        """Curvilinear + contourf round-trips to a non-empty PNG.

        Args:
            tmp_path: pytest temp-directory fixture.

        Test scenario:
            `coords` + `kind="contourf"` must produce a saveable
            figure that lands a non-zero-byte PNG on disk.
        """
        arr = self._arr_3x4()
        x1d = np.linspace(0.0, 10.0, 4)
        y1d = np.linspace(0.0, 5.0, 3)
        glyph = ArrayGlyph(arr, coords=(x1d, y1d))
        fig, ax = glyph.plot(kind="contourf")
        out = tmp_path / "coords_contourf.png"
        try:
            fig.savefig(out)
            assert out.exists(), "savefig must write a file"
            assert out.stat().st_size > 0, "PNG must be non-empty"
        finally:
            plt.close(fig)

    def test_coord_y_shape_mismatch_raises(self) -> None:
        """A y-coord with wrong shape raises `ValueError` (covers x_ok branch).

        Test scenario:
            `_validate_coords` validates x and y independently; a
            valid x with an invalid y must hit the y_ok branch and
            raise a descriptive error.
        """
        arr = self._arr_3x4()
        x_good = np.linspace(0.0, 10.0, 4)
        y_bad = np.linspace(0.0, 5.0, 7)
        with pytest.raises(ValueError, match=re.escape(_COORD_SHAPE_MISMATCH)):
            ArrayGlyph(arr, coords=(x_good, y_bad))


@pytest.mark.plot
class TestFacetingEdgeCases:
    """Edge-case coverage for `ArrayGlyph.facet` (Phase 2 / CLEO-4).

    Complements `TestFaceting` with degenerate stack sizes,
    coord-aware globals, robust + faceting interaction, name_dicts
    integrity, and a savefig round-trip on a 2x3 grid.
    """

    @staticmethod
    def _stack(n: int = 4, h: int = 6, w: int = 6, seed: int = 1337) -> np.ndarray:
        """Build a deterministic 3-D stack `(n, h, w)` for facet tests.

        Args:
            n: First-axis length.
            h: Height (penultimate axis).
            w: Width (last axis).
            seed: `np.random.default_rng` seed for reproducibility.

        Returns:
            Float `(n, h, w)` uniform-random array.
        """
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 1.0, size=(n, h, w))

    def test_single_facet_n1(self) -> None:
        """A 3-D stack with `N=1` produces a single 1x1 grid.

        Test scenario:
            Degenerate single-frame stack: shape `(1, H, W)` must
            yield `axes.shape == (1, 1)` and a single `name_dict`.
        """
        stack = self._stack(n=1)
        result = ArrayGlyph(stack).facet(col="t")
        try:
            assert result.axes.shape == (
                1,
                1,
            ), f"single facet must yield (1, 1); got {result.axes.shape}"
            assert len(result.name_dicts) == 1, "expected one name_dict"
        finally:
            plt.close(result.fig)

    def test_wide_grid_with_col_wrap(self) -> None:
        """`N=20` with `col_wrap=4` lays out a 5x4 grid.

        Test scenario:
            Confirms the integer-ceiling rows computation for a wide
            stack and that all 20 panels remain visible.
        """
        stack = self._stack(n=20, h=4, w=4)
        result = ArrayGlyph(stack).facet(col="t", col_wrap=4)
        try:
            assert result.axes.shape == (
                5,
                4,
            ), f"expected 5x4 grid; got {result.axes.shape}"
            visible = [ax for ax in result.axes.ravel() if ax.get_visible()]
            assert (
                len(visible) == 20
            ), f"all 20 panels must be visible; got {len(visible)}"
        finally:
            plt.close(result.fig)

    def test_col_wrap_larger_than_n(self) -> None:
        """`col_wrap` larger than N still works (single-row layout).

        Test scenario:
            With `N=3` and `col_wrap=10`: nrows = ceil(3/10) = 1,
            ncols = 10; 3 visible + 7 hidden trailing slots.
        """
        stack = self._stack(n=3)
        result = ArrayGlyph(stack).facet(col="t", col_wrap=10)
        try:
            assert result.axes.shape == (
                1,
                10,
            ), f"expected 1x10 grid; got {result.axes.shape}"
            visible = [ax for ax in result.axes.ravel() if ax.get_visible()]
            hidden = [ax for ax in result.axes.ravel() if not ax.get_visible()]
            assert len(visible) == 3, f"expected 3 visible; got {len(visible)}"
            assert len(hidden) == 7, f"expected 7 hidden; got {len(hidden)}"
        finally:
            plt.close(result.fig)

    def test_invalid_col_wrap_zero_raises(self) -> None:
        """`col_wrap=0` raises `ValueError` (positive-int guard).

        Test scenario:
            `col_wrap` must be a positive int; zero or negative
            values raise.
        """
        stack = self._stack(n=4)
        with pytest.raises(ValueError, match="positive int"):
            ArrayGlyph(stack).facet(col="t", col_wrap=0)

    def test_invalid_col_wrap_negative_raises(self) -> None:
        """A negative `col_wrap` is rejected."""
        stack = self._stack(n=4)
        with pytest.raises(ValueError, match="positive int"):
            ArrayGlyph(stack).facet(col="t", col_wrap=-2)

    def test_invalid_col_wrap_type_raises(self) -> None:
        """A non-int `col_wrap` is rejected (string).

        Test scenario:
            `col_wrap` must be int-typed; strings hit the
            `isinstance` guard.
        """
        stack = self._stack(n=4)
        with pytest.raises(ValueError, match="positive int"):
            ArrayGlyph(stack).facet(col="t", col_wrap="three")

    def test_col_with_2d_array_raises(self) -> None:
        """Faceting a 2-D array on `col` alone raises `ValueError`.

        Test scenario:
            Faceting on `col` requires a 3-D `(N, H, W)` stack;
            a 2-D input must raise a shape error.
        """
        arr2d = np.zeros((5, 5))
        with pytest.raises(ValueError, match="3-D array"):
            ArrayGlyph(arr2d).facet(col="t")

    def test_row_without_col_raises(self) -> None:
        """Passing `row` without `col` raises `ValueError`.

        Test scenario:
            The col+row branch requires both names; `row="lev"` alone
            is rejected.
        """
        stack = self._stack(n=4)
        with pytest.raises(ValueError, match="`col` as well"):
            ArrayGlyph(stack).facet(row="lev")

    def test_row_with_3d_arr_raises(self) -> None:
        """Faceting on row+col with a 3-D arr raises `ValueError`.

        Test scenario:
            The 4-D branch demands a 4-D stack; 3-D input must
            surface a shape error.
        """
        stack = self._stack(n=4)
        with pytest.raises(ValueError, match="4-D array"):
            ArrayGlyph(stack).facet(col="t", row="lev")

    def test_col_coords_length_mismatch_raises(self) -> None:
        """`col_coords` whose length differs from N raises `ValueError`."""
        stack = self._stack(n=4)
        with pytest.raises(ValueError, match="`col_coords` length"):
            ArrayGlyph(stack).facet(col="t", col_coords=[0, 1, 2])

    def test_col_coords_length_mismatch_4d_raises(self) -> None:
        """`col_coords` length wrong on a 4-D stack raises."""
        rng = np.random.default_rng(1337)
        stack = rng.uniform(0.0, 1.0, size=(2, 3, 4, 4))
        with pytest.raises(ValueError, match="`col_coords` length"):
            ArrayGlyph(stack).facet(col="t", row="lev", col_coords=[0])

    def test_row_coords_length_mismatch_raises(self) -> None:
        """`row_coords` whose length differs from Nrow raises."""
        rng = np.random.default_rng(1337)
        stack = rng.uniform(0.0, 1.0, size=(2, 3, 4, 4))
        with pytest.raises(ValueError, match="`row_coords` length"):
            ArrayGlyph(stack).facet(
                col="t", row="lev", col_coords=[0, 1], row_coords=[0]
            )

    def test_shared_vmin_vmax_global_min_max(self) -> None:
        """Stack-wide `vmin`/`vmax` reflect the *global* min/max across frames.

        Test scenario:
            Build two frames with disjoint ranges; the shared colorbar
            limits must equal the global `(min, max)` of the whole
            stack, not any single frame.
        """
        frame_lo = np.linspace(0.0, 1.0, 16).reshape(4, 4)
        frame_hi = np.linspace(50.0, 100.0, 16).reshape(4, 4)
        stack = np.stack([frame_lo, frame_hi], axis=0)
        result = ArrayGlyph(stack).facet(col="t")
        try:
            first = result.axes.ravel()[0].get_images()[0]
            assert first.norm.vmin == pytest.approx(
                0.0
            ), f"global vmin must be 0; got {first.norm.vmin}"
            assert first.norm.vmax == pytest.approx(
                100.0
            ), f"global vmax must be 100; got {first.norm.vmax}"
        finally:
            plt.close(result.fig)

    def test_empty_col_coords_falls_back_to_index(self) -> None:
        """`col_coords=None` titles use the integer panel index.

        Test scenario:
            With no coord list, subplot titles encode the integer
            facet index (`t=0`, `t=1`, ...).
        """
        stack = self._stack(n=3)
        result = ArrayGlyph(stack).facet(col="t", col_coords=None)
        try:
            titles = [ax.get_title() for ax in result.axes.ravel()]
            assert "t=0" in titles[0], f"expected 't=0'; got {titles[0]!r}"
            assert "t=2" in titles[-1], f"expected 't=2'; got {titles[-1]!r}"
        finally:
            plt.close(result.fig)

    def test_timestamp_col_coords(self) -> None:
        """String / timestamp `col_coords` are forwarded into titles.

        Test scenario:
            Time-axis coords often arrive as `str` (or
            `datetime`); the facet title must include the coord
            value verbatim.
        """
        stack = self._stack(n=3)
        coords = ["2024-01", "2024-02", "2024-03"]
        result = ArrayGlyph(stack).facet(col="month", col_coords=coords)
        try:
            titles = [ax.get_title() for ax in result.axes.ravel()]
            for want, title in zip(coords, titles):
                assert want in title, f"expected {want!r} in {title!r}"
            assert result.name_dicts[1] == {"month": "2024-02"}
        finally:
            plt.close(result.fig)

    def test_name_dicts_only_for_rendered_panels(self) -> None:
        """`name_dicts` length matches the number of rendered panels.

        Test scenario:
            With `N=5` and `col_wrap=3`, 5 panels are rendered
            (one slot hidden); `name_dicts` must have length 5,
            not 6.
        """
        stack = self._stack(n=5)
        result = ArrayGlyph(stack).facet(col="t", col_wrap=3)
        try:
            assert (
                len(result.name_dicts) == 5
            ), f"name_dicts must match rendered count; got {len(result.name_dicts)}"
        finally:
            plt.close(result.fig)

    def test_name_dicts_4d_contents(self) -> None:
        """Each 4-D `name_dict` carries both `col` and `row` keys.

        Test scenario:
            For a 4-D facet, every dict has both keys; with explicit
            coords, the values come from the coord sequence.
        """
        rng = np.random.default_rng(1337)
        stack = rng.uniform(0.0, 1.0, size=(2, 2, 4, 4))
        result = ArrayGlyph(stack).facet(
            col="t", row="lev", col_coords=["A", "B"], row_coords=[10, 20]
        )
        try:
            assert (
                len(result.name_dicts) == 4
            ), f"expected 4 panels; got {len(result.name_dicts)}"
            for nd in result.name_dicts:
                assert set(nd.keys()) == {
                    "t",
                    "lev",
                }, f"every name_dict must carry both keys; got {nd}"
            assert result.name_dicts[0] == {
                "t": "A",
                "lev": 10,
            }, f"first panel must use col[0]/row[0]; got {result.name_dicts[0]}"
        finally:
            plt.close(result.fig)

    def test_robust_with_faceting_global(self) -> None:
        """`robust=True` + faceting uses a global percentile over the stack.

        Test scenario:
            Per-frame robust would yield different colour limits per
            subplot. The shared global path should compute a single
            `(vmin, vmax)` over the whole stack.
        """
        rng = np.random.default_rng(1337)
        stack = rng.uniform(0.0, 1.0, size=(3, 5, 5))
        stack[0, 0, 0] = 1e6
        result = ArrayGlyph(stack).facet(col="t", robust=True)
        try:
            first = result.axes.ravel()[0].get_images()[0]
            for ax in result.axes.ravel():
                ims = ax.get_images()
                assert (
                    ims[0].norm.vmin == first.norm.vmin
                ), f"vmin must be shared; got {ims[0].norm.vmin} vs {first.norm.vmin}"
                assert (
                    ims[0].norm.vmax == first.norm.vmax
                ), f"vmax must be shared; got {ims[0].norm.vmax} vs {first.norm.vmax}"
        finally:
            plt.close(result.fig)

    def test_center_with_faceting_global(self) -> None:
        """`center` + faceting symmetrises limits over the whole stack.

        Test scenario:
            Pass `center=0` to `facet` for a stack spanning negative
            and positive values; every subplot must share a symmetric
            `(vmin, vmax)` around 0.
        """
        rng = np.random.default_rng(1337)
        stack = rng.uniform(-2.0, 4.0, size=(3, 5, 5))
        result = ArrayGlyph(stack).facet(col="t", center=0.0)
        try:
            first = result.axes.ravel()[0].get_images()[0]
            for ax in result.axes.ravel():
                ims = ax.get_images()
                assert (
                    ims[0].norm.vmin == first.norm.vmin
                ), f"vmin not shared across subplots: {ims[0].norm.vmin}"
                assert (
                    ims[0].norm.vmax == first.norm.vmax
                ), f"vmax not shared across subplots: {ims[0].norm.vmax}"
            assert first.norm.vmin == pytest.approx(
                -first.norm.vmax
            ), "centring around 0 must yield symmetric limits"
        finally:
            plt.close(result.fig)

    def test_facet_2x3_savefig_roundtrip(self, tmp_path) -> None:
        """A 2x3 facet grid round-trips to a non-empty PNG via savefig.

        Args:
            tmp_path: pytest temp-directory fixture.

        Test scenario:
            Explicit 2x3 layout (N=6, col_wrap=3) saves to disk and the
            resulting PNG has non-zero size.
        """
        stack = self._stack(n=6)
        result = ArrayGlyph(stack).facet(col="t", col_wrap=3)
        out = tmp_path / "facet_2x3.png"
        try:
            result.fig.savefig(out)
            assert out.exists(), "savefig must write the file"
            assert out.stat().st_size > 0, "PNG must be non-empty"
        finally:
            plt.close(result.fig)

    def test_facet_explicit_figsize(self) -> None:
        """An explicit `figsize` overrides the default panel-based sizing.

        Test scenario:
            `facet(figsize=(12, 4))` must produce a figure whose
            dimensions reflect the caller's choice.
        """
        stack = self._stack(n=3)
        result = ArrayGlyph(stack).facet(col="t", figsize=(12.0, 4.0))
        try:
            w, h = result.fig.get_size_inches()
            assert w == pytest.approx(12.0), f"width must be 12; got {w}"
            assert h == pytest.approx(4.0), f"height must be 4; got {h}"
        finally:
            plt.close(result.fig)

    def test_facet_with_extent_propagates_to_subplots(self) -> None:
        """A parent `extent` propagates to each sub-glyph during faceting.

        Test scenario:
            When the parent ArrayGlyph carries an `extent`, the facet
            path must convert from matplotlib's stored order back to
            the constructor's `[xmin, ymin, xmax, ymax]` and forward
            it to every per-subplot ArrayGlyph. The rendered images
            must show that extent on every subplot.
        """
        stack = self._stack(n=3)
        extent = [0.0, 0.0, 10.0, 5.0]
        result = ArrayGlyph(stack, extent=extent).facet(col="t")
        try:
            for ax in result.axes.ravel():
                if ax.get_visible():
                    images = ax.get_images()
                    assert images, "every visible subplot must have an AxesImage"
                    got = list(images[0].get_extent())
                    assert got == [
                        extent[0],
                        extent[2],
                        extent[1],
                        extent[3],
                    ], f"subplot extent must match parent; got {got}"
        finally:
            plt.close(result.fig)

    def test_facet_masked_stack_uses_compressed(self) -> None:
        """A masked-array facet input routes through `arr.compressed()`.

        Test scenario:
            When the parent ArrayGlyph builds a masked array (via
            `exclude_value`), faceting computes the shared vmin/vmax
            from `compressed()` rather than the raw view.
        """
        rng = np.random.default_rng(1337)
        stack = rng.uniform(0.0, 1.0, size=(3, 4, 4))
        stack[:, 0, 0] = -999.0
        glyph = ArrayGlyph(stack, exclude_value=[-999.0])
        result = glyph.facet(col="t")
        try:
            first = result.axes.ravel()[0].get_images()[0]
            assert (
                first.norm.vmin > -999.0
            ), "masked values must be excluded from global vmin"
        finally:
            plt.close(result.fig)

    def test_facet_all_nan_stack_rejected_at_construction(self) -> None:
        """An all-NaN stack fails at `ArrayGlyph(...)` (M2), before `facet`.

        Test scenario:
            With no finite values and no explicit `vmin`/`vmax` the
            constructor cannot derive a colour range, so it raises a clear
            `ValueError` rather than producing a NaN range that breaks
            later. `facet` is therefore never reached on such input.
        """
        nan_stack = np.full((2, 3, 3), np.nan)
        with pytest.raises(ValueError, match="no finite values"):
            ArrayGlyph(nan_stack).facet(col="t")

    def test_facet_all_nan_stack_with_explicit_limits(self) -> None:
        """Explicit `vmin`/`vmax` let an all-NaN stack be faceted anyway.

        Test scenario:
            Passing colour limits up front bypasses the M2 guard, so a
            placeholder all-NaN stack still lays out and the shared
            `(vmin, vmax)` is the one the caller supplied.
        """
        nan_stack = np.full((2, 3, 3), np.nan)
        result = ArrayGlyph(nan_stack, vmin=0.0, vmax=1.0).facet(col="t")
        try:
            for ax in result.axes.ravel():
                im = ax.get_images()[0]
                assert im.norm.vmin == pytest.approx(0.0)
                assert im.norm.vmax == pytest.approx(1.0)
        finally:
            plt.close(result.fig)

    def test_facet_returns_array_of_axes(self) -> None:
        """`FacetGrid.axes` is always a 2-D ndarray (`squeeze=False`).

        Test scenario:
            Even for a 1xN layout, `axes` is a 2-D `ndarray`, never
            a 1-D vector or bare `Axes`.
        """
        stack = self._stack(n=4)
        result = ArrayGlyph(stack).facet(col="t")
        try:
            assert isinstance(result.axes, np.ndarray), "axes must be ndarray"
            assert result.axes.ndim == 2, f"axes must be 2-D; got {result.axes.ndim}"
        finally:
            plt.close(result.fig)


@pytest.mark.plot
class TestFacetingCrossFeature:
    """Cross-cutting tests combining facet + coords / kind dispatch."""

    @staticmethod
    def _stack4d() -> np.ndarray:
        """Build a deterministic 4-D `(2, 2, 4, 5)` stack."""
        rng = np.random.default_rng(1337)
        return rng.uniform(0.0, 1.0, size=(2, 2, 4, 5))

    def test_facet_4d_with_curvilinear_coords(self) -> None:
        """Faceting a 4-D stack with shared 2-D coords forwards to every subplot.

        Test scenario:
            `coords=(x_2d, y_2d)` on a 4-D facet routes every panel
            through `pcolormesh` (the curvilinear path).
        """
        stack = self._stack4d()
        h, w = stack.shape[-2:]
        x1d = np.linspace(0.0, 10.0, w)
        y1d = np.linspace(0.0, 5.0, h)
        x2d, y2d = np.meshgrid(x1d, y1d)
        glyph = ArrayGlyph(stack, coords=(x2d, y2d))
        result = glyph.facet(col="t", row="lev", kind="pcolormesh")
        try:
            for ax in result.axes.ravel():
                if ax.get_visible():
                    assert (
                        len(ax.collections) >= 1
                    ), "every visible panel must hold a QuadMesh"
                    assert (
                        len(ax.get_images()) == 0
                    ), "pcolormesh path must not produce AxesImages"
        finally:
            plt.close(result.fig)

    def test_facet_then_savefig_no_error(self, tmp_path) -> None:
        """Calling savefig on a facet result completes without error.

        Test scenario:
            Smoke test confirming the facet API surface yields a
            saveable figure suitable for downstream report pipelines.
        """
        rng = np.random.default_rng(1337)
        stack = rng.uniform(0.0, 1.0, size=(4, 5, 5))
        result = ArrayGlyph(stack).facet(col="t")
        out = tmp_path / "facet_smoke.png"
        try:
            result.fig.savefig(out)
            assert out.exists(), "expected facet PNG to land on disk"
        finally:
            plt.close(result.fig)


@pytest.mark.plot
class TestAnimateDataGetterEdgeCases:
    """Edge-case coverage for the `animate(data_getter=...)` callback (Phase 2 / CLEO-7).

    Complements `TestAnimateDataGetter` with masked-array
    return values, callback exceptions, dtype divergence, and the
    interaction with `display_cell_value=False`.
    """

    @staticmethod
    def _stack(n: int = 4, h: int = 5, w: int = 5) -> np.ndarray:
        """Build a deterministic `(n, h, w)` stack for animate tests."""
        rng = np.random.default_rng(1337)
        return rng.uniform(0.0, 1.0, size=(n, h, w))

    def test_data_getter_positional_arg_signature(self) -> None:
        """`data_getter` is invoked with a single positional `int` arg.

        Test scenario:
            The callback contract uses positional invocation
            (`data_getter(i)`); a getter that asserts on positional
            int input must not see kwargs.
        """
        stack = self._stack(n=3)
        observed: list = []

        def getter(i):
            observed.append(i)
            assert isinstance(i, int), f"i must be int; got {type(i)}"
            return stack[i]

        glyph = ArrayGlyph(stack[0])
        anim = glyph.animate(time=list(range(3)), data_getter=getter)
        try:
            assert isinstance(anim, FuncAnimation), "animate must return FuncAnimation"
            assert observed, "getter must be invoked at least once during init"
            assert all(
                isinstance(x, int) for x in observed
            ), f"all calls must be positional ints; got {observed}"
        finally:
            plt.close(glyph.fig)

    def test_data_getter_returns_masked_array(self) -> None:
        """A getter returning a masked array still renders.

        Test scenario:
            Masked arrays are common upstream of NetCDF readers;
            `np.asarray` strips the mask in `_fetch_frame` and the
            animation must complete without raising.
        """
        stack = self._stack(n=3)
        mask = np.zeros_like(stack[0], dtype=bool)
        mask[0, 0] = True

        def getter(i):
            return np.ma.array(stack[i], mask=mask)

        glyph = ArrayGlyph(stack[0])
        anim = glyph.animate(time=list(range(3)), data_getter=getter)
        try:
            assert isinstance(anim, FuncAnimation), "masked-array frames must animate"
        finally:
            plt.close(glyph.fig)

    def test_data_getter_returns_nan_array(self) -> None:
        """A getter returning NaN-laden frames renders without error."""
        stack = self._stack(n=3)
        stack[0, 0, 0] = np.nan

        def getter(i):
            frame = stack[i].copy()
            frame[0, 0] = np.nan
            return frame

        glyph = ArrayGlyph(stack[0])
        anim = glyph.animate(time=list(range(3)), data_getter=getter)
        try:
            assert isinstance(anim, FuncAnimation), "NaN frames must animate"
        finally:
            plt.close(glyph.fig)

    def test_data_getter_raises_propagates(self, tmp_path) -> None:
        """An exception inside `data_getter` propagates to the caller.

        Args:
            tmp_path: pytest temp-directory fixture.

        Test scenario:
            The first frame is fetched eagerly in `animate` for shape
            validation; a getter that raises on `i=0` must surface
            the original `RuntimeError` unchanged.
        """
        glyph = ArrayGlyph(np.zeros((5, 5)))

        def bad_getter(i):
            raise RuntimeError(f"boom at i={i}")

        with pytest.raises(RuntimeError, match="boom at i=0"):
            glyph.animate(time=list(range(3)), data_getter=bad_getter)

    def test_data_getter_called_per_time_entry(self, tmp_path) -> None:
        """`n_frames` equals `len(time)` when `data_getter` is set.

        Args:
            tmp_path: pytest temp-directory fixture.

        Test scenario:
            With a `time` list longer than the underlying stack, the
            callback governs the frame count; rendering the saved GIF
            must invoke the getter for every time entry.
        """
        stack = self._stack(n=4)
        calls: list = []

        def getter(i):
            calls.append(i)
            return stack[i % stack.shape[0]]

        glyph = ArrayGlyph(stack[0])
        anim = glyph.animate(time=list(range(6)), data_getter=getter)
        out = tmp_path / "long_time.gif"
        try:
            glyph.save_animation(str(out), fps=2)
            unique_indices = set(calls)
            assert (
                max(unique_indices) >= 5
            ), f"getter must see the trailing time index; saw {sorted(unique_indices)}"
        finally:
            plt.close(anim._fig)

    def test_data_getter_dtype_cast(self) -> None:
        """A getter returning a different dtype is silently coerced.

        Test scenario:
            The first frame is `float64`; a getter returning
            `float32` must not raise — `np.asarray` produces a
            compatible view for `im.set_data`.
        """
        stack = self._stack(n=3).astype(np.float64)

        def getter(i):
            return stack[i].astype(np.float32)

        glyph = ArrayGlyph(stack[0])
        anim = glyph.animate(time=list(range(3)), data_getter=getter)
        try:
            assert isinstance(anim, FuncAnimation), "dtype mismatch must not raise"
        finally:
            plt.close(glyph.fig)

    def test_data_getter_with_display_cell_value_false(self) -> None:
        """`data_getter` + `display_cell_value=False` works (default path).

        Test scenario:
            The pre-existing `display_cell_value=True` interaction is
            known-broken with a lazy callback; this confirms the
            default `False` path still functions.
        """
        stack = self._stack(n=3)
        glyph = ArrayGlyph(stack[0])
        anim = glyph.animate(
            time=list(range(3)),
            data_getter=lambda i: stack[i],
            display_cell_value=False,
        )
        try:
            assert isinstance(
                anim, FuncAnimation
            ), "data_getter + display_cell_value=False must animate"
        finally:
            plt.close(glyph.fig)

    def test_data_getter_explicit_vmin_vmax(self) -> None:
        """`data_getter` honours explicit `vmin` / `vmax`."""
        stack = self._stack(n=3)
        glyph = ArrayGlyph(stack[0])
        anim = glyph.animate(
            time=list(range(3)),
            data_getter=lambda i: stack[i],
            vmin=-0.5,
            vmax=1.5,
        )
        try:
            assert (
                glyph.default_options["vmin"] == -0.5
            ), f"vmin must be -0.5; got {glyph.default_options['vmin']}"
            assert (
                glyph.default_options["vmax"] == 1.5
            ), f"vmax must be 1.5; got {glyph.default_options['vmax']}"
        finally:
            plt.close(glyph.fig)

    def test_data_getter_inconsistent_shape_after_first(self, tmp_path) -> None:
        """A getter that changes shape on a later frame raises inside `_fetch_frame`.

        Args:
            tmp_path: pytest temp-directory fixture.

        Test scenario:
            The first frame validates at construction time; the
            per-frame shape guard in `_fetch_frame` catches a callback
            that misbehaves on frame `i>=1`. Saving the GIF forces
            execution of the closure and surfaces the error.
        """
        stack = self._stack(n=3)
        glyph = ArrayGlyph(stack[0])

        def getter(i):
            if i == 0:
                return stack[0]
            return np.zeros((99, 99))

        anim = glyph.animate(time=list(range(3)), data_getter=getter)
        out = tmp_path / "bad_shape.gif"
        try:
            with pytest.raises(ValueError, match="data_getter` returned shape"):
                glyph.save_animation(str(out), fps=2)
        finally:
            plt.close(anim._fig)

    def test_data_getter_explicit_text_loc(self) -> None:
        """A user-supplied `text_loc` skips the default-init branch.

        Test scenario:
            Default `text_loc=None` is rewritten to an axes-fraction anchor;
            passing an explicit value covers the alternate branch where the
            rewrite is skipped and data coordinates are used instead.
        """
        stack = self._stack(n=3)
        glyph = ArrayGlyph(stack[0])
        anim = glyph.animate(
            time=list(range(3)),
            data_getter=lambda i: stack[i],
            text_loc=[0.05, 0.05],
        )
        try:
            assert isinstance(
                anim, FuncAnimation
            ), "explicit text_loc must produce an animation"
            assert glyph._day_text.get_transform() is glyph.ax.transData
        finally:
            plt.close(glyph.fig)

    def test_default_text_loc_label_stays_inside_axes(self) -> None:
        """The default frame label must not clip past the axes bounds.

        Test scenario:
            Regression test for the inverted-Y-axis clipping bug: with no
            explicit `text_loc`, the label's rendered bounding box must sit
            fully within the axes bounding box on both axes, for a square
            and a moderately rectangular array shape. Does not cover an
            extremely narrow axes (e.g. a 20-column array), where the
            label can still overflow horizontally simply because the
            available width is smaller than the label's rendered text at
            the default font size — a physical-space limit no anchor
            choice can fix; callers with very narrow images should pass
            an explicit `text_loc` or a smaller `cbar_label_size`.
        """
        for shape in [(4, 48, 48), (4, 100, 50)]:
            arr = np.zeros(shape)
            glyph = ArrayGlyph(arr, figsize=(4, 4))
            try:
                glyph.animate(time=list(range(shape[0])), interval=150)
                glyph._day_text.set_text("Date = 2024-01-01")
                glyph.fig.canvas.draw()
                renderer = glyph.fig.canvas.get_renderer()
                label_bbox = glyph._day_text.get_window_extent(renderer=renderer)
                axes_bbox = glyph.ax.get_window_extent(renderer=renderer)
                assert label_bbox.y1 <= axes_bbox.y1, f"label clips above axes for {shape}"
                assert label_bbox.y0 >= axes_bbox.y0, f"label clips below axes for {shape}"
                assert label_bbox.x1 <= axes_bbox.x1, f"label clips right of axes for {shape}"
                assert label_bbox.x0 >= axes_bbox.x0, f"label clips left of axes for {shape}"
            finally:
                plt.close(glyph.fig)

    def test_data_getter_background_color_threshold_set(self) -> None:
        """Setting `background_color_threshold` exercises the if-branch.

        Test scenario:
            The default-None branch computes the threshold from the
            frame; an explicit value must go through `im.norm(...)`
            directly. This covers the `if ... is not None` path of
            the lazy animate body.
        """
        stack = self._stack(n=3)
        glyph = ArrayGlyph(stack[0])
        anim = glyph.animate(
            time=list(range(3)),
            data_getter=lambda i: stack[i],
            background_color_threshold=0.5,
        )
        try:
            assert isinstance(
                anim, FuncAnimation
            ), "explicit threshold must not break the animate path"
        finally:
            plt.close(glyph.fig)


class TestMappableAndColorbarToggle:
    """Tests for the exposed mappable (`self.im`) and the `add_colorbar` toggle.

    Covers issues #128 (suppress the built-in colorbar for shared-axes
    composition) and #129 (expose the colour-mapped artist for a uniform
    `(glyph, mappable)` contract). Exercises both the `plot` and `animate`
    render paths, the RGB branch, and the `_merge_kwargs` validation gate.
    """

    @staticmethod
    def _sample_arr() -> np.ndarray:
        """Return a small 2-D array suitable for every kind.

        Returns:
            np.ndarray: A 5x5 float array of monotonically increasing values.
        """
        return np.arange(25, dtype=float).reshape(5, 5)

    @staticmethod
    def _stack(n: int = 3, h: int = 5, w: int = 5) -> np.ndarray:
        """Return a 3-D stack of frames for the `animate` path.

        Args:
            n: Number of frames.
            h: Frame height.
            w: Frame width.

        Returns:
            np.ndarray: An ``(n, h, w)`` float array.
        """
        return np.arange(n * h * w, dtype=float).reshape(n, h, w)

    @staticmethod
    def _rgb_arr() -> np.ndarray:
        """Return a deterministic 3-band array for the RGB branch.

        Returns:
            np.ndarray: A ``(3, 8, 8)`` float32 array in the 0-254 range.
        """
        rng = np.random.default_rng(seed=1337)
        return rng.integers(0, 255, size=(3, 8, 8)).astype(np.float32)

    # ---- #129: the mappable is exposed via `self.im` ---------------------

    def test_im_is_none_before_plot(self):
        """`self.im` exists and is None before the first render.

        Test scenario:
            A freshly constructed glyph has the public `im` attribute
            initialised to None, so callers can always reach it.
        """
        glyph = ArrayGlyph(self._sample_arr())
        assert glyph.im is None, f"im should start as None, got {glyph.im!r}"

    @pytest.mark.parametrize(
        "kind, expected_type_path",
        [
            ("auto", "matplotlib.image.AxesImage"),
            ("imshow", "matplotlib.image.AxesImage"),
            ("pcolormesh", "matplotlib.collections.QuadMesh"),
            ("contour", "matplotlib.contour.QuadContourSet"),
            ("contourf", "matplotlib.contour.QuadContourSet"),
        ],
    )
    def test_im_type_per_kind(self, kind, expected_type_path):
        """`self.im` holds the artist matching the render kind.

        Args:
            kind: The `kind=` value passed to `plot`.
            expected_type_path: Dotted import path of the expected artist type.

        Test scenario:
            Every non-RGB kind stores its colour-mapped artist on `self.im`
            (AxesImage for imshow/auto, QuadMesh for pcolormesh,
            QuadContourSet for contour/contourf).
        """
        import importlib

        module_name, _, type_name = expected_type_path.rpartition(".")
        expected_type = getattr(importlib.import_module(module_name), type_name)

        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind=kind)
        try:
            assert isinstance(glyph.im, expected_type), (
                f"kind={kind!r} should store {type_name} on im, "
                f"got {type(glyph.im).__name__}"
            )
        finally:
            plt.close(fig)

    def test_im_is_real_artist_on_axes_imshow(self):
        """For imshow, `self.im` is the live artist on the axes, not a copy.

        Test scenario:
            The stored mappable must be the same object matplotlib placed in
            `ax.images`, so a host can drive a colorbar from it.
        """
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="imshow")
        try:
            assert glyph.im in ax.get_images(), "im is not the artist on the axes"
        finally:
            plt.close(fig)

    def test_im_is_real_artist_on_axes_pcolormesh(self):
        """For pcolormesh, `self.im` is the live `QuadMesh` in `ax.collections`.

        Test scenario:
            The stored mappable is the collection matplotlib registered on the
            axes, confirming no detached object is handed back.
        """
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="pcolormesh")
        try:
            assert glyph.im in ax.collections, "im is not the artist on the axes"
        finally:
            plt.close(fig)

    def test_im_set_for_rgb(self):
        """The RGB branch also exposes its `AxesImage` and draws no colorbar.

        Test scenario:
            RGB compositing stores the image artist on `self.im` while leaving
            `self.cbar` as None (RGB has no scalar mapping to a colorbar).
        """
        from matplotlib.image import AxesImage

        glyph = ArrayGlyph(self._rgb_arr(), rgb=[0, 1, 2])
        fig, ax = glyph.plot(kind="imshow")
        try:
            assert isinstance(
                glyph.im, AxesImage
            ), f"RGB im should be AxesImage, got {type(glyph.im).__name__}"
            assert glyph.cbar is None, "RGB path must not create a colorbar"
        finally:
            plt.close(fig)

    def test_im_supports_host_owned_colorbar(self):
        """A host can build its own colorbar from `self.im` after opting out.

        Test scenario:
            With `add_colorbar=False` the glyph draws no colorbar, but the
            exposed mappable is sufficient for the host to call
            `fig.colorbar(glyph.im, ...)` — the shared-axes composition goal.
        """
        from matplotlib.colorbar import Colorbar

        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="imshow", add_colorbar=False)
        try:
            host_cbar = fig.colorbar(glyph.im, ax=ax)
            assert isinstance(
                host_cbar, Colorbar
            ), "host should be able to build a colorbar from glyph.im"
        finally:
            plt.close(fig)

    def test_im_refreshed_on_replot(self):
        """Re-rendering updates `self.im` to the new artist.

        Test scenario:
            Calling `plot` a second time (different kind) replaces the stored
            mappable rather than leaving a stale handle from the first call.
        """
        from matplotlib.collections import QuadMesh
        from matplotlib.image import AxesImage

        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="imshow")
        try:
            assert isinstance(glyph.im, AxesImage)
            glyph.plot(kind="pcolormesh")
            assert isinstance(
                glyph.im, QuadMesh
            ), "im should reflect the most recent render kind"
        finally:
            plt.close(fig)

    def test_animate_sets_im(self):
        """`animate` stores the first-frame mappable on `self.im`.

        Test scenario:
            The animate path mirrors `plot`: after building the animation the
            glyph exposes the colour-mapped artist via `self.im`.
        """
        from matplotlib.image import AxesImage

        glyph = ArrayGlyph(self._stack())
        anim = glyph.animate(list(range(3)))
        try:
            assert isinstance(
                glyph.im, AxesImage
            ), f"animate should set im to AxesImage, got {type(glyph.im).__name__}"
        finally:
            plt.close(anim._fig)

    # ---- #128: the built-in colorbar can be suppressed -------------------

    def test_add_colorbar_accepted_without_validation_error(self):
        """`add_colorbar` passes `_merge_kwargs` and is recorded in options.

        Test scenario:
            Because the toggle was added to DEFAULT_OPTIONS, passing it does
            not trip the unknown-kwarg `ValueError`, and the value is stored
            in `default_options` for the render to read.
        """
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="imshow", add_colorbar=False)
        try:
            assert (
                glyph.default_options["add_colorbar"] is False
            ), "add_colorbar should be merged into default_options"
        finally:
            plt.close(fig)

    def test_colorbar_drawn_by_default(self):
        """Default behaviour is unchanged: a colorbar is created.

        Test scenario:
            With no `add_colorbar` kwarg the glyph still draws its colorbar,
            which adds a second Axes to the figure.
        """
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="imshow")
        try:
            assert glyph.cbar is not None, "default plot should create a colorbar"
            assert (
                len(fig.axes) == 2
            ), f"colorbar should add a second axes, got {len(fig.axes)}"
        finally:
            plt.close(fig)

    @pytest.mark.parametrize("kind", ["imshow", "pcolormesh", "contourf"])
    def test_add_colorbar_false_draws_no_colorbar(self, kind):
        """`add_colorbar=False` leaves `self.cbar is None` and adds no axes.

        Args:
            kind: The render kind under test.

        Test scenario:
            Across raster and filled-contour kinds, opting out draws no
            colorbar (no extra axes) yet still exposes the mappable.
        """
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind=kind, add_colorbar=False)
        try:
            assert glyph.cbar is None, f"kind={kind!r}: cbar should be None"
            assert (
                len(fig.axes) == 1
            ), f"kind={kind!r}: no colorbar axes expected, got {len(fig.axes)}"
            assert glyph.im is not None, f"kind={kind!r}: mappable should still be set"
        finally:
            plt.close(fig)

    def test_shared_axes_two_layers_no_colorbars(self):
        """Two ArrayGlyph layers with add_colorbar=False compose cleanly.

        Test scenario:
            A contourf field plus a contour overlay on one shared axes produce
            no per-glyph colorbars (no extra axes), and each layer exposes its
            own mappable for the host to aggregate.
        """
        fig, ax = plt.subplots()
        try:
            g1 = ArrayGlyph(self._sample_arr(), ax=ax, fig=fig)
            g1.plot(kind="contourf", add_colorbar=False)
            g2 = ArrayGlyph(self._sample_arr(), ax=ax, fig=fig)
            g2.plot(kind="contour", add_colorbar=False)

            assert g1.cbar is None and g2.cbar is None, "no per-glyph colorbars"
            assert (
                len(fig.axes) == 1
            ), f"shared axes should stay single, got {len(fig.axes)} axes"
            assert (
                g1.im is not None and g2.im is not None
            ), "each layer must expose its mappable for aggregation"
        finally:
            plt.close(fig)

    @pytest.mark.parametrize("add_colorbar", [True, False])
    def test_rgb_never_draws_colorbar(self, add_colorbar):
        """The RGB branch ignores `add_colorbar` and never draws a colorbar.

        Args:
            add_colorbar: Toggle value under test.

        Test scenario:
            RGB has no scalar mapping, so `self.cbar` stays None regardless of
            the toggle, while `self.im` is always populated.
        """
        glyph = ArrayGlyph(self._rgb_arr(), rgb=[0, 1, 2])
        fig, ax = glyph.plot(kind="imshow", add_colorbar=add_colorbar)
        try:
            assert glyph.cbar is None, "RGB path must never create a colorbar"
            assert glyph.im is not None, "RGB path must still expose the mappable"
        finally:
            plt.close(fig)

    def test_animate_add_colorbar_false(self):
        """`animate(add_colorbar=False)` draws no colorbar but exposes the mappable.

        Test scenario:
            The toggle is honoured on the animate path too: `self.cbar` is None
            and no colorbar axes is added, while `self.im` is still set.
        """
        glyph = ArrayGlyph(self._stack())
        anim = glyph.animate(list(range(3)), add_colorbar=False)
        try:
            assert glyph.cbar is None, "animate add_colorbar=False should skip cbar"
            assert glyph.im is not None, "animate should still expose the mappable"
            assert (
                len(anim._fig.axes) == 1
            ), f"no colorbar axes expected, got {len(anim._fig.axes)}"
        finally:
            plt.close(anim._fig)


class TestPlotAxAndTitleParams:
    """Tests for `ax=` and `title=` on `ArrayGlyph.plot` (issue #130).

    Verifies parity with the other glyphs: `plot(ax=)` composes onto the
    given axes (deriving its figure), `plot(title=)` sets the title, and the
    axes-resolution priority is plot(ax=) > constructor ax > fresh figure.
    """

    @staticmethod
    def _arr() -> np.ndarray:
        """Return a small 2-D array."""
        return np.arange(25, dtype=float).reshape(5, 5)

    def test_plot_ax_renders_onto_supplied_axes(self):
        """`plot(ax=ax)` draws on the given axes without making a new figure.

        Test scenario:
            A glyph constructed with no axes renders onto a caller's axes when
            it is passed to `plot`, and binds that axes' figure.
        """
        fig, ax = plt.subplots()
        try:
            glyph = ArrayGlyph(self._arr())
            out_fig, out_ax = glyph.plot(ax=ax, kind="imshow")
            assert out_ax is ax, "plot(ax=) must render onto the supplied axes"
            assert out_fig is ax.get_figure(), "figure must be derived from the axes"
            assert glyph.ax is ax, "glyph.ax should be bound to the supplied axes"
        finally:
            plt.close(fig)

    def test_plot_ax_overrides_constructor_axes(self):
        """`plot(ax=)` wins over an axes bound at construction.

        Test scenario:
            Resolution priority: an explicit `plot(ax=)` takes precedence over
            the constructor-supplied axes.
        """
        ctor_fig, ctor_ax = plt.subplots()
        plot_fig, plot_ax = plt.subplots()
        try:
            glyph = ArrayGlyph(self._arr(), ax=ctor_ax, fig=ctor_fig)
            out_fig, out_ax = glyph.plot(ax=plot_ax, kind="imshow")
            assert out_ax is plot_ax, "plot(ax=) must override the constructor axes"
            assert out_ax is not ctor_ax, "constructor axes must not win over plot(ax=)"
        finally:
            plt.close(ctor_fig)
            plt.close(plot_fig)

    def test_plot_uses_constructor_axes_when_no_plot_ax(self):
        """Without `plot(ax=)`, the constructor axes is used (no fresh figure).

        Test scenario:
            Resolution priority middle tier: a glyph built with `ax=`/`fig=`
            renders there when `plot` is called without an `ax`.
        """
        fig, ax = plt.subplots()
        try:
            glyph = ArrayGlyph(self._arr(), ax=ax, fig=fig)
            out_fig, out_ax = glyph.plot(kind="imshow")
            assert (
                out_ax is ax
            ), "constructor axes should be used when plot(ax=) omitted"
            assert out_fig is fig, "constructor figure should be reused"
        finally:
            plt.close(fig)

    def test_plot_creates_figure_when_no_axes_anywhere(self):
        """With no axes from either source, `plot` creates its own figure/axes.

        Test scenario:
            Resolution priority fallback: neither constructor nor plot supplies
            an axes, so a fresh figure/axes is created.
        """
        glyph = ArrayGlyph(self._arr())
        out_fig, out_ax = glyph.plot(kind="imshow")
        try:
            assert isinstance(out_fig, Figure), "a fresh Figure should be created"
            assert out_ax is not None, "a fresh axes should be created"
        finally:
            plt.close(out_fig)

    def test_plot_title_sets_axes_title(self):
        """`plot(title=...)` sets the axes title.

        Test scenario:
            The convenience `title` parameter is equivalent to the `title`
            option and is applied to the rendered axes.
        """
        fig, ax = plt.subplots()
        try:
            glyph = ArrayGlyph(self._arr())
            _, out_ax = glyph.plot(ax=ax, title="My Field", kind="imshow")
            assert (
                out_ax.get_title() == "My Field"
            ), f"title should be set, got {out_ax.get_title()!r}"
        finally:
            plt.close(fig)

    def test_plot_title_overrides_constructor_title(self):
        """`plot(title=)` overrides a title set at construction.

        Test scenario:
            When a title was given via constructor options, an explicit
            `plot(title=)` takes precedence.
        """
        fig, ax = plt.subplots()
        try:
            glyph = ArrayGlyph(self._arr(), title="ctor title")
            _, out_ax = glyph.plot(ax=ax, title="plot title", kind="imshow")
            assert (
                out_ax.get_title() == "plot title"
            ), f"plot title should win, got {out_ax.get_title()!r}"
        finally:
            plt.close(fig)

    def test_construct_with_ax_only_keeps_axes(self):
        """`ArrayGlyph(arr, ax=ax)` (no fig) now renders onto that axes.

        Test scenario:
            Regression for the old gate that dropped a constructor `ax` when
            `fig` was omitted; the axes must now be honoured by `plot()`.
        """
        fig, ax = plt.subplots()
        try:
            glyph = ArrayGlyph(self._arr(), ax=ax)
            _, out_ax = glyph.plot(kind="imshow")
            assert out_ax is ax, "ax-only construction must be honoured by plot()"
        finally:
            plt.close(fig)


class TestConstantValueArray:
    """Constant-value (flat) arrays must render without dividing by zero (#61/#4)."""

    @pytest.mark.parametrize("kind", ["imshow", "pcolormesh", "contourf", "contour"])
    def test_flat_array_plots_without_error(self, kind):
        """A constant field renders for every kind (no ZeroDivisionError).

        Args:
            kind: The render kind under test.

        Test scenario:
            `vmax == vmin` made `ticks_spacing` zero and the tick math raised;
            the guard now lets a flat array plot for every kind. Line `contour`
            additionally skips its (degenerate, empty) colorbar.
        """
        glyph = ArrayGlyph(np.full((10, 10), 5.0))
        fig, ax = glyph.plot(kind=kind)
        try:
            assert isinstance(fig, Figure), f"kind={kind!r} should render a figure"
            if kind == "contour":
                assert glyph.cbar is None, "degenerate contour should skip the colorbar"
        finally:
            plt.close(fig)

    def test_all_zero_array_plots(self):
        """An all-zero array (vmax==vmin==0) also renders without error.

        Test scenario:
            Boundary of the degenerate case where the value itself is zero.
        """
        glyph = ArrayGlyph(np.zeros((5, 5)))
        fig, ax = glyph.plot(kind="imshow")
        try:
            assert isinstance(fig, Figure), "all-zero array should render"
        finally:
            plt.close(fig)

    def test_constant_contour_warns_and_skips_colorbar(self):
        """A constant-field line `contour` warns and draws no colorbar.

        Test scenario:
            The degenerate contour set cannot back a line colorbar; the glyph
            warns and leaves `self.cbar` None instead of crashing.
        """
        glyph = ArrayGlyph(np.full((10, 10), 5.0))
        with pytest.warns(UserWarning, match="no contour lines"):
            fig, ax = glyph.plot(kind="contour")
        try:
            assert glyph.cbar is None, "degenerate contour should skip the colorbar"
        finally:
            plt.close(fig)

    def test_constant_contourf_does_not_warn(self):
        """A constant-field filled `contourf` draws normally without warning.

        Test scenario:
            The degenerate-contour warning is specific to line `contour`;
            `contourf` fills a region and keeps its colorbar.
        """
        import warnings as _warnings

        glyph = ArrayGlyph(np.full((10, 10), 5.0))
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            fig, ax = glyph.plot(kind="contourf")
        try:
            assert glyph.cbar is not None, "contourf should still draw a colorbar"
        finally:
            plt.close(fig)


class TestScaleToRgbPerBand:
    """Per-band percentile stretch option on `scale_to_rgb` (#1 enhancement)."""

    def test_global_default_unchanged(self):
        """The default global-max path is unchanged and does not mutate input.

        Test scenario:
            Backward-compatibility: `per_band=False` keeps the legacy scaling
            and leaves the source array untouched.
        """
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        glyph = ArrayGlyph(arr.copy())
        before = glyph.arr.copy()
        out = glyph.scale_to_rgb()
        assert out.dtype == np.uint8, f"expected uint8, got {out.dtype}"
        assert out[0].tolist() == [28, 56, 85], f"global scaling changed: {out[0]}"
        assert np.array_equal(
            glyph.arr, before
        ), "scale_to_rgb must not mutate self.arr"

    def test_per_band_stretches_each_band_to_full_range(self):
        """`per_band=True` maps each band independently across 0-255.

        Test scenario:
            A 3-band stack with differing per-band ranges comes back uint8 with
            each band spanning the full 0-255 range after the percentile cut.
        """
        rng = np.random.default_rng(0)
        stack = rng.uniform(10.0, 200.0, size=(8, 8, 3))
        out = ArrayGlyph(np.zeros((4, 4))).scale_to_rgb(stack, per_band=True)
        assert out.shape == (8, 8, 3) and out.dtype == np.uint8, "shape/dtype wrong"
        for band in range(3):
            assert int(out[..., band].min()) == 0, f"band {band} min should be 0"
            assert int(out[..., band].max()) == 255, f"band {band} max should be 255"

    def test_per_band_does_not_mutate_input(self):
        """The per-band path leaves the source array unmodified.

        Test scenario:
            Purity — the caller's stack is unchanged after stretching.
        """
        stack = np.random.default_rng(1).uniform(0, 100, size=(5, 5, 3))
        before = stack.copy()
        ArrayGlyph(np.zeros((4, 4))).scale_to_rgb(stack, per_band=True)
        assert np.array_equal(stack, before), "input stack must not be mutated"

    def test_per_band_requires_3d(self):
        """`per_band=True` on a non-3-D array raises a clear ValueError.

        Test scenario:
            A 2-D array has no band axis, so per-band stretching is rejected.
        """
        with pytest.raises(ValueError, match="3-D"):
            ArrayGlyph(np.zeros((4, 4))).scale_to_rgb(np.zeros((4, 4)), per_band=True)

    def test_all_zero_global_no_zero_division(self):
        """Global scaling of an all-zero array returns zeros, not a crash.

        Test scenario:
            `arr.max() == 0` previously divided by zero; the guard returns an
            all-zero uint8 array instead.
        """
        out = ArrayGlyph(np.zeros((3, 3))).scale_to_rgb()
        assert out.dtype == np.uint8 and int(out.max()) == 0, "expected all-zero uint8"

    def test_per_band_all_nan_band_is_zero_filled_without_warning(self):
        """An all-NaN (or flat) band yields a zero band and emits no warning.

        Test scenario:
            `nanpercentile` on an all-NaN band returns NaN cuts; the guard
            detects the non-finite range, emits a flat zero band, and the
            `RuntimeWarning` is suppressed (treated as an error here to prove
            it does not surface).
        """
        import warnings as _warnings

        stack = np.full((4, 4, 3), np.nan)
        stack[..., 1] = 5.0  # band 1 is a flat (no-range) band
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            out = ArrayGlyph(np.zeros((4, 4))).scale_to_rgb(stack, per_band=True)
        assert out.dtype == np.uint8, "expected uint8 output"
        assert int(out[..., 0].max()) == 0, "all-NaN band should be zero-filled"
        assert int(out[..., 1].max()) == 0, "flat band has no range -> zero"

    def test_per_band_partial_nan_band_stretches_finite_values(self):
        """A band with some NaNs stretches its finite values across 0-255.

        Test scenario:
            NaN pixels do not break the stretch; the finite part still spans
            the full output range (NaN pixels are not asserted on, only that
            the band has real contrast and the call does not raise).
        """
        import warnings as _warnings

        rng = np.random.default_rng(3)
        stack = rng.uniform(10.0, 200.0, size=(6, 6, 3))
        stack[0, 0, 0] = np.nan  # a single NaN pixel in band 0
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")  # no stray cast warning may surface
            out = ArrayGlyph(np.zeros((4, 4))).scale_to_rgb(stack, per_band=True)
        assert out.dtype == np.uint8, "expected uint8 output"
        assert (
            int(out[..., 0].max()) == 255
        ), "finite values should still stretch to 255"
        assert int(out[0, 0, 0]) == 0, "the NaN pixel should map to 0"


class TestArrayGlyphHillshade:
    """Tests for the `hillshade` relief-shading option (regular-grid DEMs)."""

    @staticmethod
    def _dem():
        yy, xx = np.mgrid[0:30, 0:40]
        return 10.0 + 0.5 * yy + 200.0 * np.exp(
            -(((xx - 30) / 6) ** 2 + ((yy - 15) / 8) ** 2)
        ) + 8.0 * np.sin(xx / 4.0)

    def test_off_by_default_draws_scalar_image(self):
        """Without hillshade the imshow image carries a scalar array."""
        _, ax = ArrayGlyph(self._dem(), cmap="terrain").plot()
        assert ax.images[0].get_array().ndim == 2
        plt.close("all")

    def test_hillshade_draws_rgba_image(self):
        """`hillshade=True` replaces the image data with a shaded RGBA image."""
        _, ax = ArrayGlyph(self._dem(), cmap="terrain", hillshade=True).plot()
        assert ax.images[0].get_array().shape[-1] == 4
        plt.close("all")

    def test_colorbar_survives_hillshade(self):
        """The colorbar still reflects the cmap/norm after shading."""
        g = ArrayGlyph(self._dem(), cmap="terrain", hillshade=True)
        g.plot()
        assert g.cbar is not None
        lo, hi = g.cbar.mappable.get_clim()
        assert hi > lo
        plt.close("all")

    def test_hillshade_options_dict(self):
        """A dict of hillshade overrides (vert_exag, multidirectional) is honoured."""
        _, ax = ArrayGlyph(
            self._dem(), cmap="gist_earth",
            hillshade={"vert_exag": 6, "azimuth": 300, "multidirectional": 3},
        ).plot()
        assert ax.images[0].get_array().shape[-1] == 4
        plt.close("all")

    def test_hillshade_warns_on_non_imshow_kind(self):
        """`hillshade` on a kind it cannot shade warns instead of silently no-op-ing."""
        with pytest.warns(UserWarning, match="hillshade is only applied to kind='imshow'"):
            ArrayGlyph(self._dem(), cmap="terrain", hillshade=True).plot(kind="pcolormesh")
        plt.close("all")
