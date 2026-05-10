import os
import shutil

import pytest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from PIL import Image

from cleopatra.array_glyph import ArrayGlyph, FacetGrid


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


@pytest.mark.plot
class TestPlotKindDispatch:
    """Tests for ``ArrayGlyph.plot(kind=...)`` dispatch.

    Covers the ``kind`` enum introduced in CLEO-1:
    ``"auto" | "imshow" | "pcolormesh" | "contour" | "contourf"``.
    """

    @staticmethod
    def _sample_arr() -> np.ndarray:
        """Return a small 2-D array suitable for every kind."""
        return np.arange(25, dtype=float).reshape(5, 5)

    def test_default_kind_is_backward_compatible(self):
        """``plot()`` with no kind kwarg still returns (fig, ax) with an image."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot()
        assert isinstance(fig, Figure)
        assert len(ax.get_images()) >= 1

    def test_kind_imshow(self):
        """``kind="imshow"`` renders via ``ax.imshow``/``matshow``."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="imshow")
        assert isinstance(fig, Figure)
        assert len(ax.get_images()) >= 1

    def test_kind_pcolormesh(self):
        """``kind="pcolormesh"`` runs and adds a ``QuadMesh`` to ``ax``."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="pcolormesh")
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1

    def test_kind_contour(self):
        """``kind="contour"`` runs and adds contour collections to ``ax``."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="contour")
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1

    def test_kind_contourf(self):
        """``kind="contourf"`` runs and adds filled contour collections."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="contourf")
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1

    def test_invalid_kind_raises(self):
        """``kind="bogus"`` raises ``ValueError`` listing the valid kinds."""
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
        """``kind="imshow", display_cell_value=True`` still draws cell text."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="imshow", display_cell_value=True)
        assert isinstance(fig, Figure)
        # Each cell of the 5x5 grid yields one Text artist; allow >= 25
        # because matplotlib also creates axis label/title Text objects.
        cell_texts = [t for t in ax.texts]
        assert len(cell_texts) >= 25

    def test_contour_skips_cell_value_silently(self):
        """``display_cell_value=True`` is skipped for ``kind="contour"``."""
        glyph = ArrayGlyph(self._sample_arr())
        fig, ax = glyph.plot(kind="contour", display_cell_value=True)
        assert isinstance(fig, Figure)
        # No per-cell text annotations should have been emitted.
        assert len(ax.texts) == 0


@pytest.mark.plot
class TestColourKwargs:
    """Tests for the xarray-aligned colour kwargs introduced in CLEO-3.

    Covers ``robust``, ``center``, ``levels``, ``extend``, and
    ``cbar_kwargs`` on :class:`ArrayGlyph`. Each kwarg is tested both in
    isolation and (where relevant) in combination with the existing
    ``color_scale`` enum and the ``kind=`` dispatch from CLEO-1.
    """

    @staticmethod
    def _arr_with_outliers() -> np.ndarray:
        """Build a 10x10 array of normal values plus extreme outliers.

        The bulk of the values sit in ``[0, 1)``; two cells hold large
        negative and positive outliers so ``robust=True`` has something
        to clip.
        """
        rng = np.random.default_rng(seed=0)
        body = rng.random(98)
        full = np.concatenate([body, [-1000.0, 1000.0]])
        return full.reshape(10, 10)

    def test_robust_clips_vmin_vmax(self):
        """``robust=True`` uses the 2nd/98th percentile, ignoring outliers."""
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
        """An explicit ``vmin``/``vmax`` always wins over ``robust=True``."""
        arr = self._arr_with_outliers()
        glyph = ArrayGlyph(arr, robust=True, vmin=0.0, vmax=1.0)
        assert glyph.vmin == 0.0
        assert glyph.vmax == 1.0

    def test_center_zero_makes_vmin_vmax_symmetric(self):
        """``center=0`` symmetrises around zero using the larger half-range."""
        arr = np.linspace(-3.0, 7.0, 30).reshape(5, 6)
        glyph = ArrayGlyph(arr, center=0.0)
        assert glyph.vmin == pytest.approx(-7.0)
        assert glyph.vmax == pytest.approx(7.0)

    def test_center_switches_default_cmap_to_rdbu_r(self):
        """``center=...`` without an explicit cmap selects ``RdBu_r``."""
        arr = np.linspace(-2.0, 2.0, 16).reshape(4, 4)
        glyph = ArrayGlyph(arr, center=0.0)
        assert glyph.default_options["cmap"] == "RdBu_r"

    def test_center_does_not_override_explicit_cmap(self):
        """An explicit ``cmap=`` always wins over the diverging default."""
        arr = np.linspace(-2.0, 2.0, 16).reshape(4, 4)
        glyph = ArrayGlyph(arr, center=0.0, cmap="viridis")
        assert glyph.default_options["cmap"] == "viridis"

    def test_levels_int_creates_discretised_norm(self):
        """``levels=N`` switches the linear norm to a ``BoundaryNorm``."""
        arr = np.linspace(0.0, 1.0, 100).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        fig, ax = glyph.plot(kind="imshow", levels=5)
        assert isinstance(fig, Figure)
        images = ax.get_images()
        assert images, "expected at least one AxesImage from imshow"
        norm = images[0].norm
        assert isinstance(norm, BoundaryNorm), (
            f"levels=int should switch to BoundaryNorm, got {type(norm)}"
        )
        assert len(norm.boundaries) == 5

    def test_levels_list_uses_explicit_edges(self):
        """``levels=[...]`` is forwarded as explicit bin edges."""
        arr = np.linspace(0.0, 1.0, 100).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        edges = [0.0, 0.25, 0.5, 0.75, 1.0]
        fig, ax = glyph.plot(kind="imshow", levels=edges)
        assert isinstance(fig, Figure)
        norm = ax.get_images()[0].norm
        assert isinstance(norm, BoundaryNorm)
        np.testing.assert_array_almost_equal(norm.boundaries, edges)

    def test_extend_both_when_levels_set_and_extend_none(self):
        """``levels=[...], extend=None`` auto-resolves ``extend`` to ``both``."""
        arr = np.linspace(0.0, 1.0, 100).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        edges = [0.0, 0.5, 1.0]
        fig, ax = glyph.plot(kind="imshow", levels=edges)
        # The cbar is attached to the image's mappable; resolve via
        # the glyph attribute we expose at create_color_bar time.
        cbar = glyph.cbar
        assert cbar.extend == "both"

    def test_extend_explicit_overrides_auto(self):
        """An explicit ``extend="min"`` is preserved despite ``levels``."""
        arr = np.linspace(0.0, 1.0, 100).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        edges = [0.0, 0.5, 1.0]
        fig, ax = glyph.plot(kind="imshow", levels=edges, extend="min")
        assert glyph.cbar.extend == "min"

    def test_cbar_kwargs_merge_overrides_defaults(self):
        """``cbar_kwargs`` merges over the defaults; user keys win."""
        arr = np.linspace(0.0, 1.0, 100).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        fig, ax = glyph.plot(
            kind="imshow",
            cbar_kwargs={"label": "Custom", "shrink": 0.5},
        )
        cbar = glyph.cbar
        # The label set via cbar_kwargs flows through ``set_label`` and
        # surfaces on the underlying ``YAxis`` label artist.
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
        """``robust=True, center=0`` clips outliers then symmetrises."""
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
        """``robust=True, kind="pcolormesh"`` renders a QuadMesh."""
        arr = self._arr_with_outliers()
        glyph = ArrayGlyph(arr, robust=True)
        fig, ax = glyph.plot(kind="pcolormesh")
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1


@pytest.mark.plot
class TestValidateExtend:
    """Tests for :py:meth:`ArrayGlyph._validate_extend` static method."""

    def test_none_is_accepted(self):
        """``extend=None`` returns silently (auto-resolve at render time)."""
        result = ArrayGlyph._validate_extend(None)
        assert result is None, f"Expected None, got {result!r}"

    @pytest.mark.parametrize(
        "value", ["neither", "both", "min", "max"]
    )
    def test_allowed_values_accepted(self, value):
        """Each allowed string returns silently.

        Args:
            value: A valid ``extend`` keyword.
        """
        ArrayGlyph._validate_extend(value)

    @pytest.mark.parametrize("bogus", ["always", "MIN", "", "neither2", "extend"])
    def test_invalid_string_raises(self, bogus):
        """Any other string raises ``ValueError`` listing the allowed set.

        Args:
            bogus: An invalid extend value to reject.
        """
        with pytest.raises(ValueError, match="Invalid extend"):
            ArrayGlyph._validate_extend(bogus)

    def test_invalid_extend_in_constructor_raises(self):
        """Passing ``extend='banana'`` to ``__init__`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="Invalid extend"):
            ArrayGlyph(np.arange(9).reshape(3, 3), extend="banana")

    def test_invalid_extend_in_plot_raises(self):
        """Passing ``extend='banana'`` to ``plot`` raises ``ValueError``."""
        glyph = ArrayGlyph(np.arange(9).reshape(3, 3))
        with pytest.raises(ValueError, match="Invalid extend"):
            glyph.plot(extend="banana")


@pytest.mark.plot
class TestRobustLimits:
    """Tests for :py:meth:`ArrayGlyph._robust_limits` static method."""

    def test_basic_array(self):
        """A plain array returns 2nd / 98th percentile floats."""
        rng = np.random.default_rng(1337)
        arr = rng.normal(size=10000)
        vmin_r, vmax_r = ArrayGlyph._robust_limits(arr)
        assert vmin_r > arr.min(), "Robust vmin should exceed full min"
        assert vmax_r < arr.max(), "Robust vmax should fall under full max"

    def test_all_nan_raises(self):
        """An all-NaN array surfaces as ``ValueError``."""
        arr = np.full((5, 5), np.nan)
        with pytest.raises(ValueError, match="no finite values"):
            ArrayGlyph._robust_limits(arr)

    def test_all_inf_raises(self):
        """An all-Inf array (no finite cells) surfaces as ``ValueError``."""
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
    """Tests for :py:meth:`ArrayGlyph._center_limits` static method."""

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


class TestPrepareArrayValidation:
    """Tests for ``ArrayGlyph.prepare_array`` and RGB constructor validation."""

    def test_too_few_bands_raises(self):
        """An RGB array with fewer than 3 bands raises ``ValueError``."""
        arr = np.zeros((2, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="3 arrays"):
            ArrayGlyph(arr, rgb=[0, 1])

    def test_prepare_array_with_cutoff_only(self):
        """``cutoff`` is applied via the surface-reflectance branch."""
        arr = np.random.default_rng(0).integers(
            0, 10000, size=(3, 5, 5)
        ).astype(np.float32)
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
        """``_prepare_sentinel_rgb`` returns clipped data with no cutoff path."""
        arr = np.random.default_rng(0).integers(
            0, 10000, size=(3, 5, 5)
        ).astype(np.float32)
        glyph = ArrayGlyph(np.zeros((1, 1)))
        result = glyph.prepare_array(
            arr, rgb=[0, 1, 2], surface_reflectance=10000
        )
        assert result.shape == (5, 5, 3)
        assert np.all((0.0 <= result) & (result <= 1.0))


class TestPlotImGetCbarKwInvalidKind:
    """Tests for :py:meth:`ArrayGlyph._plot_im_get_cbar_kw` invalid-kind branch."""

    def test_invalid_kind_raises(self):
        """An unsupported kind in the helper raises ``ValueError``.

        Test scenario:
            ``ArrayGlyph.plot`` validates the kind before calling
            ``_plot_im_get_cbar_kw``; reaching the helper directly with
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
    """Tests for the ``arr`` property setter."""

    def test_setter_replaces_array(self):
        """Assigning to ``arr`` swaps out the internal masked array."""
        a = np.arange(9).reshape(3, 3)
        glyph = ArrayGlyph(a)
        new = np.zeros((4, 4))
        glyph.arr = new
        assert glyph.arr is new, "Setter should store the new array reference"


class TestExcludeValueProperty:
    """Tests for the ``exclude_value`` getter/setter."""

    def test_default_is_nan(self):
        """The default ``exclude_value`` is ``np.nan``."""
        glyph = ArrayGlyph(np.arange(9).reshape(3, 3))
        # exclude_value comparison: nan is nan via identity
        assert glyph.exclude_value is np.nan or (
            isinstance(glyph.exclude_value, float)
            and np.isnan(glyph.exclude_value)
        )

    def test_setter_updates_value(self):
        """Setter updates the stored exclude value."""
        glyph = ArrayGlyph(np.arange(9).reshape(3, 3))
        glyph.exclude_value = -9999
        assert glyph.exclude_value == -9999


@pytest.mark.plot
class TestPlotRecomputeBranch:
    """Tests for the recompute-on-plot path in :py:meth:`ArrayGlyph.plot`."""

    def test_passing_robust_in_plot_recomputes_limits(self):
        """``plot(robust=True)`` recomputes vmin/vmax from the percentile path."""
        rng = np.random.default_rng(1337)
        body = rng.random(98)
        arr = np.concatenate([body, [-1000.0, 1000.0]]).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        before_vmin = glyph.vmin
        glyph.plot(robust=True)
        assert glyph.vmin > before_vmin, (
            "Robust pass on plot should clip the lower outlier"
        )
        assert glyph.vmax < 1000.0, (
            "Robust pass on plot should clip the upper outlier"
        )

    def test_passing_center_in_plot_switches_cmap(self):
        """``plot(center=0)`` selects ``RdBu_r`` when no explicit cmap is given."""
        arr = np.linspace(-3.0, 5.0, 25).reshape(5, 5)
        glyph = ArrayGlyph(arr)
        glyph.plot(center=0.0)
        assert glyph.default_options["cmap"] == "RdBu_r"

    def test_passing_center_in_plot_with_explicit_cmap_keeps_user_choice(self):
        """``plot(center=0, cmap='viridis')`` keeps the explicit cmap."""
        arr = np.linspace(-3.0, 5.0, 25).reshape(5, 5)
        glyph = ArrayGlyph(arr)
        glyph.plot(center=0.0, cmap="viridis")
        assert glyph.default_options["cmap"] == "viridis"

    def test_passing_explicit_vmin_vmax_in_plot_overrides_constructor(self):
        """``plot(vmin=, vmax=)`` overrides constructor-derived limits."""
        arr = np.arange(25, dtype=float).reshape(5, 5)
        glyph = ArrayGlyph(arr)
        glyph.plot(vmin=5.0, vmax=15.0)
        assert glyph.vmin == 5.0
        assert glyph.vmax == 15.0

    def test_invalid_kwarg_raises(self):
        """Plotting with an unknown kwarg raises ``ValueError``."""
        glyph = ArrayGlyph(np.arange(9).reshape(3, 3))
        with pytest.raises(ValueError, match="not correct"):
            glyph.plot(banana_count=12)

    def test_recompute_with_explicit_ticks_spacing(self):
        """``plot(robust=True, ticks_spacing=...)`` keeps the user's spacing."""
        rng = np.random.default_rng(1337)
        body = rng.random(98)
        arr = np.concatenate([body, [-1000.0, 1000.0]]).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        glyph.plot(robust=True, ticks_spacing=0.1)
        assert glyph.default_options["ticks_spacing"] == 0.1, (
            "Explicit ticks_spacing must win over the recompute path"
        )


@pytest.mark.plot
class TestPlotIdempotenceAndPurity:
    """Tests confirming repeated plot calls remain safe."""

    def test_replot_same_kind_does_not_raise(self):
        """Calling ``plot(kind="contour")`` twice produces equivalent state."""
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
    """Edge cases for :py:meth:`ArrayGlyph.animate`."""

    def test_invalid_kwarg_raises(self, coello_data: np.ndarray, animate_time_list: list):
        """An unknown kwarg to ``animate`` raises ``ValueError``."""
        glyph = ArrayGlyph(coello_data)
        with pytest.raises(ValueError, match="not correct"):
            glyph.animate(animate_time_list, banana=42)

    def test_explicit_vmin_vmax_in_animate(
        self, coello_data: np.ndarray, animate_time_list: list, no_data_value: float
    ):
        """``animate(vmin=, vmax=)`` overrides constructor-derived limits."""
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
        """``animate(points=...)`` adds scatter overlays without raising."""
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
        """``animate(ticks_spacing=...)`` overrides the auto-computed spacing."""
        glyph = ArrayGlyph(coello_data, exclude_value=[no_data_value])
        anim = glyph.animate(animate_time_list, ticks_spacing=50.0)
        assert anim is not None
        assert glyph.default_options["ticks_spacing"] == 50.0


class TestExcludeValueMultiValue:
    """Cover the ``len(exclude_value) > 1`` constructor branch."""

    def test_two_exclude_values_masks_both(self):
        """Two exclude values should mask both target cells via ``logical_or``."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        glyph = ArrayGlyph(arr, exclude_value=[1.0, 9.0])
        assert glyph.arr.mask[0, 0], "Cell with value 1 must be masked"
        assert glyph.arr.mask[2, 2], "Cell with value 9 must be masked"
        assert not glyph.arr.mask[1, 1], "Middle cell must remain unmasked"


class TestPrepareArrayPercentilePath:
    """Cover the ``percentile`` branch in ``prepare_array``."""

    def test_percentile_drives_scaling(self):
        """``percentile=2`` triggers the ``scale_percentile`` branch."""
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
        """``contour`` with a non-None norm and integer levels uses both paths."""
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
        """``contourf`` with explicit edges uses the ``level_edges`` branch."""
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
        """``color_scale='midpoint'`` pre-fills the array; contour skips the mask branch."""
        arr = np.linspace(-1.0, 1.0, 100).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        fig, ax = glyph.plot(kind="contour", color_scale="midpoint", midpoint=0.0)
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1


@pytest.mark.plot
class TestCenterCmapDoesNotApplyToRgb:
    """RGB compositing path skips the center/cmap auto-switch logic."""

    def test_rgb_arr_renders_without_norm(self):
        """An RGB array renders via ``imshow`` and ignores center logic."""
        rgb_arr = np.random.default_rng(0).integers(
            0, 255, size=(3, 8, 8)
        ).astype(np.float32)
        glyph = ArrayGlyph(rgb_arr, rgb=[0, 1, 2])
        fig, ax = glyph.plot()
        assert isinstance(fig, Figure)
        # No colorbar is created on the RGB path; ``cbar`` is unset.
        assert not hasattr(glyph, "cbar"), (
            "RGB compositing should not create a colorbar"
        )


@pytest.mark.plot
class TestCoordsCurvilinear:
    """Tests for curvilinear / non-uniform ``coords=(x, y)`` support (CLEO-2).

    Covers the ``coords`` constructor kwarg introduced in C-2, the
    automatic ``kind="auto"`` -> ``pcolormesh`` dispatch when coords
    are present, mutual exclusivity with ``extent``, and shape
    validation.
    """

    @staticmethod
    def _arr_3x4() -> np.ndarray:
        """Small ``(3 rows, 4 cols)`` test array."""
        return np.arange(12, dtype=float).reshape(3, 4)

    def test_1d_coords_auto_routes_to_pcolormesh(self):
        """1-D ``coords`` + ``kind="auto"`` renders as a ``QuadMesh``."""
        arr = self._arr_3x4()
        x = np.linspace(0.0, 10.0, 4)
        y = np.linspace(0.0, 5.0, 3)
        glyph = ArrayGlyph(arr, coords=(x, y))
        fig, ax = glyph.plot(kind="auto")
        assert isinstance(fig, Figure)
        # ``pcolormesh`` produces a QuadMesh that lives in ax.collections.
        assert len(ax.collections) >= 1
        # And not an image (imshow would have been the wrong path).
        assert len(ax.get_images()) == 0

    def test_2d_meshgrid_coords_render_via_pcolormesh(self):
        """2-D ``(X, Y)`` from ``np.meshgrid`` renders via ``pcolormesh``."""
        arr = self._arr_3x4()
        x1d = np.linspace(0.0, 10.0, 4)
        y1d = np.linspace(0.0, 5.0, 3)
        x2d, y2d = np.meshgrid(x1d, y1d)
        glyph = ArrayGlyph(arr, coords=(x2d, y2d))
        fig, ax = glyph.plot(kind="pcolormesh")
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1

    def test_imshow_with_coords_raises(self):
        """``kind="imshow"`` together with ``coords`` raises ``ValueError``.

        Document the design choice: imshow renders on the array's
        index grid and cannot honour arbitrary (x, y) coordinate
        arrays. Callers must use ``kind="pcolormesh"`` or
        ``kind="auto"`` instead.
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
        with pytest.raises(ValueError, match="doesn't match data shape"):
            ArrayGlyph(arr, coords=(x_bad, y))

    def test_extent_and_coords_together_raises(self):
        """Passing both ``extent`` and ``coords`` raises ``ValueError``."""
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
        """``coords`` must be a length-2 sequence; otherwise ``TypeError``."""
        arr = self._arr_3x4()
        with pytest.raises(TypeError, match="length-2 sequence"):
            ArrayGlyph(arr, coords="oops")

    def test_2d_coords_with_contourf(self):
        """Curvilinear 2-D coords also forward through ``kind="contourf"``."""
        arr = self._arr_3x4()
        x1d = np.linspace(0.0, 10.0, 4)
        y1d = np.linspace(0.0, 5.0, 3)
        x2d, y2d = np.meshgrid(x1d, y1d)
        glyph = ArrayGlyph(arr, coords=(x2d, y2d))
        fig, ax = glyph.plot(kind="contourf")
        assert isinstance(fig, Figure)
        assert len(ax.collections) >= 1

    def test_backward_compat_no_coords_renders_imshow(self):
        """An ``ArrayGlyph(arr)`` with no coords still renders via imshow."""
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
    """Tests for the faceting API introduced in CLEO-4 (``ArrayGlyph.facet``)."""

    @staticmethod
    def _stack_3d(n: int = 4, h: int = 10, w: int = 10) -> np.ndarray:
        """Build a 3-D ``(n, h, w)`` stack of distinct frames."""
        rng = np.random.default_rng(seed=42)
        return rng.uniform(0.0, 1.0, size=(n, h, w))

    @staticmethod
    def _stack_4d(
        n_col: int = 2, n_row: int = 3, h: int = 10, w: int = 10
    ) -> np.ndarray:
        """Build a 4-D ``(n_col, n_row, h, w)`` stack."""
        rng = np.random.default_rng(seed=43)
        return rng.uniform(0.0, 1.0, size=(n_col, n_row, h, w))

    def test_3d_col_only_builds_single_row(self):
        """3-D stack with ``col="t"`` yields a 1xN grid of subplots."""
        stack = self._stack_3d(n=4)
        result = ArrayGlyph(stack).facet(col="t")
        assert isinstance(result, FacetGrid)
        assert result.axes.shape == (1, 4)
        # Each subplot has rendered an artist (imshow -> images).
        for ax in result.axes.ravel():
            assert len(ax.get_images()) >= 1

    def test_col_wrap_produces_wrapped_grid(self):
        """``col_wrap=3`` on 6 panels yields a 2x3 grid."""
        stack = self._stack_3d(n=6)
        result = ArrayGlyph(stack).facet(col="t", col_wrap=3)
        assert result.axes.shape == (2, 3)
        # All 6 panels visible.
        visible = [ax for ax in result.axes.ravel() if ax.get_visible()]
        assert len(visible) == 6

    def test_4d_col_row_grid(self):
        """4-D stack with ``col`` + ``row`` yields a NrowxNcol grid."""
        stack = self._stack_4d(n_col=2, n_row=3)
        result = ArrayGlyph(stack).facet(col="t", row="level")
        # nrows=3 (from n_row), ncols=2 (from n_col).
        assert result.axes.shape == (3, 2)
        assert len(result.name_dicts) == 6

    def test_shared_vmin_vmax_across_subplots(self):
        """Every subplot must share the same ``norm.vmin``/``norm.vmax``."""
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
        """The returned ``FacetGrid`` exposes a single shared colorbar."""
        stack = self._stack_3d(n=3)
        result = ArrayGlyph(stack).facet(col="t")
        # ``cbar`` is taken from the first rendered subplot; not None
        # for the non-RGB path.
        assert result.cbar is not None

    def test_coord_aware_titles(self):
        """``col_coords`` plugs into the per-subplot title and name_dicts."""
        stack = self._stack_3d(n=4)
        coords = [0, 6, 12, 18]
        result = ArrayGlyph(stack).facet(col="hour", col_coords=coords)
        # Each title must reference the coord value, not the index.
        for ax, want in zip(result.axes.ravel(), coords):
            assert str(want) in ax.get_title()
        # And ``name_dicts`` mirrors xarray's structure.
        assert result.name_dicts[0] == {"hour": 0}
        assert result.name_dicts[-1] == {"hour": 18}

    def test_no_col_no_row_raises(self):
        """Calling ``facet()`` without ``col`` or ``row`` raises ``ValueError``."""
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
        """When ``col_wrap`` does not divide N, trailing axes are hidden."""
        stack = self._stack_3d(n=5)
        result = ArrayGlyph(stack).facet(col="t", col_wrap=3)
        # Layout: 3 cols x 2 rows = 6 slots, 5 panels rendered.
        assert result.axes.shape == (2, 3)
        hidden = [ax for ax in result.axes.ravel() if not ax.get_visible()]
        assert len(hidden) == 1
