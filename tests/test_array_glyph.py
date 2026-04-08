import os
import shutil

import pytest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from PIL import Image

from cleopatra.array_glyph import ArrayGlyph


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
        assert array.no_elem == 89
        assert array.vmin == 0
        assert array.vmax == 88


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

    @pytest.mark.skipif(
        shutil.which("ffmpeg") is None, reason="FFmpeg not installed"
    )
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

    @pytest.mark.skipif(
        shutil.which("ffmpeg") is None, reason="FFmpeg not installed"
    )
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

    @pytest.mark.skipif(
        shutil.which("ffmpeg") is None, reason="FFmpeg not installed"
    )
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
