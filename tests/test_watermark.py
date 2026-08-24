"""Tests for the figure watermark / brand-mark helper -- issue #312.

Covers `cleopatra.styling.watermark.stamp_mark`: corner placement in
figure-fraction coordinates, dpi-invariant sizing, undistorted aspect, the
optional gaussian-blurred halo, image-input handling (RGBA/RGB arrays,
float arrays, file paths), and input validation.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from cleopatra.styling.watermark import _CORNERS, _HALO_SIGMAS, DEFAULT_BLUR, stamp_mark


@pytest.fixture
def fig():
    """Provide a 8x6-inch figure with one populated subplot.

    Returns:
        matplotlib.figure.Figure: A figure carrying a small imshow so the mark
        is drawn over real content.
    """
    figure = plt.figure(figsize=(8.0, 6.0))
    figure.add_subplot(111).imshow(np.zeros((10, 10)))
    yield figure
    plt.close(figure)


@pytest.fixture
def logo():
    """Provide a 40x80 (h x w) opaque white RGBA logo array.

    Returns:
        np.ndarray: A ``(40, 80, 4)`` ``uint8`` RGBA array (aspect 0.5).
    """
    arr = np.zeros((40, 80, 4), dtype=np.uint8)
    arr[..., :3] = 255
    arr[..., 3] = 255
    return arr


class TestStampMark:
    """Tests for `stamp_mark`."""

    @pytest.mark.parametrize(
        "corner, at_right, at_top",
        [
            ("lower right", True, False),
            ("lower left", False, False),
            ("upper right", True, True),
            ("upper left", False, True),
        ],
    )
    def test_places_in_each_corner(self, fig, logo, corner, at_right, at_top):
        """The mark lands in the requested corner, in figure-fraction coords.

        Args:
            fig: Figure fixture.
            logo: RGBA logo fixture.
            corner: The corner anchor under test.
            at_right: Whether that corner is on the right edge.
            at_top: Whether that corner is on the top edge.

        Test scenario:
            For each corner the returned axes' bounds match the width/height
            derived from `frac` and the image + figure aspect, offset from the
            correct edges by `margin`.
        """
        frac, margin = 0.11, 0.025
        ax = stamp_mark(
            fig, logo, frac=frac, corner=corner, margin=margin, shadow=False
        )
        x0, y0, w, h = (float(v) for v in ax.get_position().bounds)

        exp_w = frac
        exp_h = frac * (40 / 80) * (8.0 / 6.0)
        exp_x = (1.0 - margin - exp_w) if at_right else margin
        exp_y = (1.0 - margin - exp_h) if at_top else margin
        assert np.allclose([x0, y0, w, h], [exp_x, exp_y, exp_w, exp_h]), (
            f"{corner}: got {(x0, y0, w, h)}, expected {(exp_x, exp_y, exp_w, exp_h)}"
        )

    def test_size_holds_across_dpi(self, fig, logo):
        """The mark's figure-fraction position is unchanged by dpi.

        Test scenario:
            Because the mark is an inset axes in figure-fraction coordinates,
            its bounds are identical at 100 and 300 dpi -- so it stays
            proportional across an MP4 master, a web copy, and a GIF.
        """
        ax = stamp_mark(fig, logo, shadow=False)
        fig.set_dpi(100)
        at_100 = tuple(ax.get_position().bounds)
        fig.set_dpi(300)
        at_300 = tuple(ax.get_position().bounds)
        assert np.allclose(at_100, at_300), (
            f"position changed with dpi: {at_100} vs {at_300}"
        )

    def test_image_is_undistorted(self, fig, logo):
        """The on-figure mark keeps the image's aspect ratio.

        Test scenario:
            The rendered width:height in inches equals the source image's
            width:height (0.5 here), even though the figure itself is not
            square -- so a logo is never stretched.
        """
        ax = stamp_mark(fig, logo, shadow=False)
        box = ax.get_position()
        w_in = box.width * 8.0
        h_in = box.height * 6.0
        assert np.isclose(h_in / w_in, 40 / 80), (
            f"aspect distorted: {h_in / w_in} != {40 / 80}"
        )

    def test_tall_logo_stays_on_canvas(self, fig):
        """A portrait logo is sized by height and never overflows the figure.

        Test scenario:
            A 10:1 tall logo would, if `frac` only sized the width, derive a
            height above 1 and land off-canvas. `frac` instead caps the mark's
            longer (height) side, so the mark stays within ``[0, 1]`` in both
            dimensions and keeps its aspect ratio.
        """
        tall = np.zeros((400, 40, 4), dtype=np.uint8)
        tall[..., :3] = 255
        tall[..., 3] = 255
        ax = stamp_mark(fig, tall, frac=0.5, corner="upper left", shadow=False)
        x0, y0, w, h = (float(v) for v in ax.get_position().bounds)
        assert 0.0 <= x0 and x0 + w <= 1.0, f"mark overflows horizontally: {(x0, w)}"
        assert 0.0 <= y0 and y0 + h <= 1.0, f"mark overflows vertically: {(y0, h)}"
        assert np.isclose(max(w, h), 0.5), f"longer side should equal frac: {(w, h)}"
        assert np.isclose((h * 6.0) / (w * 8.0), 400 / 40), (
            f"tall logo distorted: {(w, h)}"
        )

    def test_shadow_composites_into_one_axes(self, fig, logo):
        """`shadow=True` composites the halo into the mark's own axes.

        Test scenario:
            The halo and mark are alpha-composited into a single image before
            placement, so only one axes is added -- one resample keeps the halo
            in register with the mark's soft edges, which two independently
            resampled axes would not guarantee.
        """
        n_before = len(fig.axes)
        ax = stamp_mark(fig, logo, shadow=True)
        assert len(fig.axes) - n_before == 1, (
            "halo should be composited, not drawn on its own axes"
        )
        drawn = ax.images[0].get_array()
        assert drawn.shape[2] == 4, "the composited mark should carry alpha"
        assert drawn.shape[0] > logo.shape[0] and drawn.shape[1] > logo.shape[1], (
            f"halo canvas should exceed the mark: {drawn.shape} vs {logo.shape}"
        )

    def test_halo_grows_the_axes_so_the_mark_keeps_its_size(self, fig, logo):
        """The axes rect grows by the halo pad, leaving the mark at `frac`.

        Test scenario:
            The padded canvas is ``1 + 2 * _HALO_SIGMAS * blur`` times the
            mark's width. The axes rect is grown by exactly that, so the mark
            itself still measures `frac` -- without the compensation it would
            render at 1/grow (about 72 %) of the requested size.
        """
        frac = 0.2
        plain = stamp_mark(fig, logo, frac=frac, shadow=False).get_position().width
        haloed = stamp_mark(fig, logo, frac=frac, shadow=True).get_position().width
        grow = 1.0 + 2.0 * _HALO_SIGMAS * DEFAULT_BLUR
        assert np.isclose(plain, frac), f"unhaloed mark should measure frac: {plain}"
        assert np.isclose(haloed, frac * grow, rtol=0.02), (
            f"haloed axes should be frac*{grow:.3f}={frac * grow:.4f}, got {haloed:.4f}"
        )

    def test_mark_painted_extent_is_frac(self):
        """The **mark's own painted width** is `frac` of the figure, halo or not.

        Test scenario:
            This is the invariant the padding can silently break: render the
            figure and measure the red mark's actual extent in pixels. Sizing
            the padded canvas to `frac` instead would put the visible mark at
            roughly 7.9 % for a requested 11 %, while the axes bbox still looked
            correct.
        """
        for shadow in (False, True):
            figure = plt.figure(figsize=(8.0, 6.0), facecolor="white")
            mark = np.zeros((40, 80, 4), dtype=np.uint8)
            mark[..., 0] = 255  # opaque pure red, distinct from the black halo
            mark[..., 3] = 255
            stamp_mark(
                figure, mark, frac=0.25, corner="lower left", margin=0.15, shadow=shadow
            )
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            red = (rgba[..., 0] > 200) & (rgba[..., 1] < 60) & (rgba[..., 2] < 60)
            cols = np.where(red.any(axis=0))[0]
            painted = (cols.max() - cols.min() + 1) / rgba.shape[1]
            assert abs(painted - 0.25) < 0.01, (
                f"shadow={shadow}: mark painted at {painted:.4f} of the figure, expected 0.25"
            )
            plt.close(figure)

    def test_halo_is_centred_not_offset(self):
        """The halo spreads symmetrically, with no light direction implied.

        Test scenario:
            A drop shadow offset down-and-right would put more darkening on one
            side of the mark than the other. Measure the halo's reach beyond the
            mark on the left and the right; they should match.
        """
        figure = plt.figure(figsize=(8.0, 6.0), facecolor="white")
        mark = np.zeros((40, 80, 4), dtype=np.uint8)
        mark[..., 0] = 255
        mark[..., 3] = 255
        stamp_mark(figure, mark, frac=0.3, corner="lower left", margin=0.3, shadow=True)
        figure.canvas.draw()
        rgba = np.asarray(figure.canvas.buffer_rgba())
        red = (rgba[..., 0] > 200) & (rgba[..., 1] < 60) & (rgba[..., 2] < 60)
        darkened = rgba[..., :3].min(axis=2) < 250  # halo or mark, vs the white canvas
        red_cols = np.where(red.any(axis=0))[0]
        dark_cols = np.where(darkened.any(axis=0))[0]
        left = red_cols.min() - dark_cols.min()
        right = dark_cols.max() - red_cols.max()
        assert left > 0 and right > 0, (
            f"halo should reach past both sides: {(left, right)}"
        )
        assert abs(left - right) <= 2, (
            f"halo is off-centre: {left}px left vs {right}px right"
        )
        plt.close(figure)

    def test_no_shadow_adds_single_axes(self, fig, logo):
        """`shadow=False` draws only the mark axes.

        Test scenario:
            Without a shadow exactly one axes is added.
        """
        n_before = len(fig.axes)
        stamp_mark(fig, logo, shadow=False)
        assert len(fig.axes) - n_before == 1, (
            "no-shadow stamp should add exactly one axes"
        )

    def test_returns_frameless_axes_above_content(self, fig, logo):
        """The mark axes is frameless, off, and above ordinary content.

        Test scenario:
            The returned axes has its frame off (no white box), no visible
            axis, and a very high zorder so it draws on top of the plot.
        """
        ax = stamp_mark(fig, logo, shadow=False)
        assert not ax.get_frame_on(), "mark axes must be frameless (no background box)"
        assert not ax.axison, "mark axes must have its axis turned off"
        assert ax.get_zorder() >= 1_000_000, (
            "mark must sit above ordinary figure content"
        )

    def test_unknown_corner_raises(self, fig, logo):
        """An unrecognised corner raises a clear `ValueError` naming the input.

        Test scenario:
            ``corner="middle"`` is rejected before any drawing, and the message
            names the bad value.
        """
        with pytest.raises(ValueError, match=r"corner must be one of.*'middle'"):
            stamp_mark(fig, logo, corner="middle")

    @pytest.mark.parametrize("frac", [0.0, -0.1, 1.5])
    def test_invalid_frac_raises(self, fig, logo, frac):
        """A `frac` outside ``(0, 1]`` raises `ValueError`.

        Args:
            fig: Figure fixture.
            logo: RGBA logo fixture.
            frac: The out-of-range fraction under test.

        Test scenario:
            Zero, negative, and above-one fractions are all rejected.
        """
        with pytest.raises(ValueError, match="frac must be in"):
            stamp_mark(fig, logo, frac=frac)

    def test_margin_accepts_an_xy_pair(self, fig, logo):
        """`margin` takes an ``(x, y)`` pair, not just one scalar.

        Test scenario:
            The showcase tucks the mark hard into the corner vertically while
            keeping a horizontal gap, so a scalar cannot express its placement.
            ``margin=(0.025, 0.0)`` puts the mark flush with the bottom edge and
            0.025 in from the right.
        """
        ax = stamp_mark(
            fig, logo, corner="lower right", margin=(0.025, 0.0), shadow=False
        )
        x0, y0, w, _ = (float(v) for v in ax.get_position().bounds)
        assert np.isclose(y0, 0.0), f"vertical margin 0 should sit flush: {y0}"
        assert np.isclose(x0 + w, 1.0 - 0.025), (
            f"horizontal margin should be 0.025: {x0 + w}"
        )

    @pytest.mark.parametrize("bad", ["0.1", (0.1, 0.2, 0.3), (0.1,), None])
    def test_malformed_margin_raises(self, fig, logo, bad):
        """A margin that is neither a scalar nor an ``(x, y)`` pair raises.

        Args:
            fig: Figure fixture.
            logo: RGBA logo fixture.
            bad: A malformed margin value.

        Test scenario:
            Strings, wrong-length sequences, and a non-iterable non-number
            (``None``) are all rejected with a clear message.
        """
        with pytest.raises(ValueError, match="margin must be"):
            stamp_mark(fig, logo, margin=bad)

    def test_negative_blur_raises(self, fig, logo):
        """A negative `blur` raises `ValueError`.

        Test scenario:
            Blur is a sigma, so it cannot be negative.
        """
        with pytest.raises(ValueError, match="blur must be non-negative"):
            stamp_mark(fig, logo, blur=-0.1)

    @pytest.mark.parametrize("margin", [-0.01, 1.0, 1.5])
    def test_invalid_margin_raises(self, fig, logo, margin):
        """A `margin` outside ``[0, 1)`` raises `ValueError`.

        Args:
            fig: Figure fixture.
            logo: RGBA logo fixture.
            margin: The out-of-range margin under test.

        Test scenario:
            Negative and >= 1 margins are rejected.
        """
        with pytest.raises(ValueError, match="margin must be in"):
            stamp_mark(fig, logo, margin=margin)

    def test_rgb_array_gets_opaque_alpha(self, fig):
        """A 3-channel RGB array is accepted and stamped opaque.

        Test scenario:
            An ``(H, W, 3)`` array renders without error -- an opaque alpha is
            added internally -- producing a normal mark axes.
        """
        rgb = np.full((30, 30, 3), 128, dtype=np.uint8)
        ax = stamp_mark(fig, rgb, shadow=False)
        assert ax.images, "RGB array should produce a drawn image"

    def test_float_array_accepted(self, fig):
        """A float ``0-1`` RGBA array is accepted (scaled to ``uint8``).

        Test scenario:
            A float image in ``[0, 1]`` renders without error.
        """
        rgba = np.ones((30, 60, 4), dtype=np.float32)
        ax = stamp_mark(fig, rgba, shadow=False)
        assert ax.images, "float array should produce a drawn image"

    def test_missing_path_raises(self, fig, tmp_path):
        """A non-existent file path raises `FileNotFoundError`.

        Test scenario:
            Passing a path that does not exist surfaces a clear
            `FileNotFoundError` at stamp time, not a confusing later failure.
        """
        with pytest.raises(FileNotFoundError):
            stamp_mark(fig, str(tmp_path / "does_not_exist.png"), shadow=False)

    def test_uint16_array_rejected(self, fig):
        """A non-``uint8`` integer array is rejected, not truncated mod 256.

        Test scenario:
            A ``uint16`` image (value 1000 would become 232 under a bare
            ``uint8`` cast) raises a `ValueError` naming the dtype instead of
            silently garbling the mark.
        """
        u16 = np.full((20, 20, 4), 1000, dtype=np.uint16)
        with pytest.raises(ValueError, match="uint8"):
            stamp_mark(fig, u16, shadow=False)

    def test_out_of_range_float_array_rejected(self, fig):
        """A float array outside ``[0, 1]`` is rejected, not clipped to white.

        Test scenario:
            A ``0-255`` float array (common way to hold an image) raises a
            `ValueError` about the ``[0, 1]`` contract rather than flattening to
            all-white.
        """
        f255 = np.full((20, 20, 4), 128.0, dtype=np.float32)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            stamp_mark(fig, f255, shadow=False)

    def test_nan_float_array_rejected(self, fig):
        """A float array with a ``NaN`` is rejected, not silently cast to 0.

        Test scenario:
            A ``NaN`` makes ``min``/``max`` ``NaN``, whose comparisons are all
            ``False``, so without a finiteness check it would slip past the
            range guard and cast to 0 with a `RuntimeWarning`. It must raise a
            `ValueError` about finite values instead.
        """
        arr = np.ones((20, 20, 4), dtype=np.float32)
        arr[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            stamp_mark(fig, arr, shadow=False)

    @pytest.mark.parametrize("bad", [np.zeros((10, 10)), np.zeros((10, 10, 2))])
    def test_bad_array_shape_raises(self, fig, bad):
        """A non-``(H, W, 3|4)`` array raises `ValueError`.

        Args:
            fig: Figure fixture.
            bad: An array with an unsupported shape.

        Test scenario:
            2-D and 2-channel arrays are rejected with a shape message.
        """
        with pytest.raises(ValueError, match="must be"):
            stamp_mark(fig, bad, shadow=False)

    def test_file_path_input(self, fig, tmp_path, logo):
        """A PNG file path is loaded via PIL and stamped.

        Args:
            fig: Figure fixture.
            tmp_path: pytest temp directory.
            logo: RGBA logo fixture, written to disk.

        Test scenario:
            Writing the logo to a PNG and passing its path produces the same
            placement as passing the array directly.
        """
        png = tmp_path / "logo.png"
        Image.fromarray(logo).save(png)
        ax_path = stamp_mark(fig, str(png), shadow=False)
        assert ax_path.images, "file-path input should produce a drawn image"
        assert np.allclose(
            ax_path.get_position().bounds,
            (0.865, 0.025, 0.11, 0.11 * 0.5 * (8.0 / 6.0)),
        ), f"file-path placement wrong: {ax_path.get_position().bounds}"

    def test_frac_controls_width(self, fig, logo):
        """A larger `frac` yields a proportionally wider mark.

        Test scenario:
            Doubling `frac` doubles the mark's figure-fraction width.
        """
        small = stamp_mark(fig, logo, frac=0.1, shadow=False).get_position().width
        big = stamp_mark(fig, logo, frac=0.2, shadow=False).get_position().width
        assert np.isclose(big, 2 * small), (
            f"frac should scale width linearly: {big} vs {small}"
        )

    def test_corners_constant_exposes_four_anchors(self):
        """`_CORNERS` lists exactly the four documented anchors.

        Test scenario:
            The accepted-corner set matches the documented four.
        """
        assert set(_CORNERS) == {
            "lower right",
            "lower left",
            "upper right",
            "upper left",
        }
