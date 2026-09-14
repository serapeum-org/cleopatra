"""Tests for the figure watermark helpers -- issues #312 and #365.

Covers `cleopatra.styling.watermark.stamp_mark`: corner placement in
figure-fraction coordinates, dpi-invariant sizing, undistorted aspect, the
optional gaussian-blurred halo, image-input handling (RGBA/RGB arrays,
float arrays, file paths), and input validation.

And `stamp_watermark`, its text counterpart: fraction-based sizing that means
the same thing for any text, credit-line placement by margin, the deliberate
outline asymmetry between the two artists, and input validation matching
`stamp_mark`'s.

Plus `WatermarkMixin`, which exposes both as glyph methods so a caller does not
have to import them and pass the figure by hand.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from cleopatra.glyphs.globe.textured_globe_glyph import TexturedGlobeGlyph
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
from cleopatra.glyphs.primitives.line_glyph import LineGlyph
from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph
from cleopatra.glyphs.stats.histogram_glyph import HistogramGlyph
from cleopatra.glyphs.stats.kde_glyph import KDEGlyph
from cleopatra.styling.watermark import (
    _CORNERS,
    _HALO_SIGMAS,
    DEFAULT_BLUR,
    WatermarkMixin,
    _fit_text_to_frac,
    stamp_mark,
    stamp_watermark,
)


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
        assert 0.0 <= x0, f"mark overflows the left edge: x0={x0}"
        assert x0 + w <= 1.0, f"mark overflows the right edge: {(x0, w)}"
        assert 0.0 <= y0, f"mark overflows the bottom edge: y0={y0}"
        assert y0 + h <= 1.0, f"mark overflows the top edge: {(y0, h)}"
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
        assert drawn.shape[0] > logo.shape[0], (
            f"halo canvas height should exceed the mark: {drawn.shape} vs {logo.shape}"
        )
        assert drawn.shape[1] > logo.shape[1], (
            f"halo canvas width should exceed the mark: {drawn.shape} vs {logo.shape}"
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

    def test_blur_zero_skips_the_halo(self, fig, logo):
        """`blur=0` with `shadow=True` skips the invisible halo, so no bbox growth.

        Test scenario:
            A zero-sigma halo is invisible but would still pad the canvas by a
            pixel and inflate the axes bbox. With `blur=0` the composite is
            skipped, so the axes measures exactly `frac` -- like `shadow=False`.
        """
        frac = 0.2
        haloed = (
            stamp_mark(fig, logo, frac=frac, shadow=True, blur=0.0).get_position().width
        )
        assert np.isclose(haloed, frac), (
            f"blur=0 should not grow the axes: {haloed} != {frac}"
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

    def test_undistorted_with_halo(self):
        """The mark keeps its aspect ratio even under the asymmetric halo pad.

        Test scenario:
            The halo pads by a fixed pixel count, so a non-square mark gets an
            asymmetric grow (``grow_w != grow_h``). Render with the halo and
            measure the red mark's painted width and height; their ratio must
            still equal the image's 40:80 aspect -- the mark is not stretched.
        """
        figure = plt.figure(figsize=(8.0, 6.0), facecolor="white")
        mark = np.zeros((40, 80, 4), dtype=np.uint8)
        mark[..., 0] = 255
        mark[..., 3] = 255
        stamp_mark(
            figure, mark, frac=0.25, corner="lower left", margin=0.15, shadow=True
        )
        figure.canvas.draw()
        rgba = np.asarray(figure.canvas.buffer_rgba())
        red = (rgba[..., 0] > 200) & (rgba[..., 1] < 60) & (rgba[..., 2] < 60)
        h_px = np.ptp(np.where(red.any(axis=1))[0]) + 1
        w_px = np.ptp(np.where(red.any(axis=0))[0]) + 1
        assert abs((h_px / w_px) - (40 / 80)) < 0.02, (
            f"mark distorted under the halo: h/w={h_px / w_px:.4f}, expected {40 / 80}"
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
        assert left > 0, f"halo should reach past the left side: {(left, right)}"
        assert right > 0, f"halo should reach past the right side: {(left, right)}"
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

    def test_margin_plus_size_off_canvas_raises(self, fig, logo):
        """A margin that leaves no room for the mark raises, not silent overflow.

        Test scenario:
            `frac` and `margin` are each in range, but `margin=0.95` +
            `frac=0.11` sums past 1, which would place the mark off the opposite
            edge. `stamp_mark` raises a clear `ValueError` instead.
        """
        with pytest.raises(ValueError, match="exceeds the figure"):
            stamp_mark(fig, logo, frac=0.11, margin=0.95, shadow=False)

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

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_float_array_rejected(self, fig, bad):
        """A float array with ``NaN``/``inf`` is rejected, not silently cast.

        Args:
            fig: Figure fixture.
            bad: The non-finite value to plant.

        Test scenario:
            ``NaN`` makes ``min``/``max`` ``NaN`` (all comparisons ``False``),
            and ``inf`` would fail the cast too, so without a finiteness check
            they slip past the range guard and cast with a `RuntimeWarning`. All
            three must raise a `ValueError` about finite values instead.
        """
        arr = np.ones((20, 20, 4), dtype=np.float32)
        arr[0, 0, 0] = bad
        with pytest.raises(ValueError, match="finite"):
            stamp_mark(fig, arr, shadow=False)

    def test_float_rgb_array_accepted(self, fig):
        """A float ``0-1`` **RGB** ``(H, W, 3)`` array is accepted (opaque alpha).

        Test scenario:
            Only float RGBA was covered; a float RGB array must also render, with
            an opaque alpha added internally.
        """
        rgb = np.full((30, 60, 3), 0.5, dtype=np.float32)
        ax = stamp_mark(fig, rgb, shadow=False)
        assert ax.images, "float RGB array should produce a drawn image"

    def test_bool_array_rejected(self, fig):
        """A ``bool`` array is rejected with a clear message, not silently cast.

        Test scenario:
            A bool array is not ``uint8``; it must raise the non-float/uint8
            error rather than being truncated.
        """
        b = np.ones((20, 20, 4), dtype=bool)
        with pytest.raises(ValueError, match="uint8"):
            stamp_mark(fig, b, shadow=False)

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

    def test_zero_size_image_rejected(self, fig):
        """A zero-size image dimension raises clearly, not a matplotlib error.

        Test scenario:
            A ``(0, W, C)`` array passes the shape check but has no pixels;
            `stamp_mark` rejects it up front rather than surfacing a confusing
            matplotlib-internal reduction error.
        """
        empty = np.zeros((0, 10, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="zero-size"):
            stamp_mark(fig, empty, shadow=False)

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


def _share_of_figure(fig, artist):
    """The artist's longer side as a fraction of the corresponding figure side.

    Args:
        fig: The figure the artist is on.
        artist: The text artist to measure.

    Returns:
        float: The larger of the width and height fractions.
    """
    figure_box = fig.get_window_extent()
    box = artist.get_window_extent()
    return float(max(box.width / figure_box.width, box.height / figure_box.height))


def _notebook_helper(fig, text, *, angle=30, text_alpha=0.65):
    """The earthlens notebooks' local helper, reproduced for comparison.

    Args:
        fig: The figure to stamp.
        text: The brand text.
        angle: Rotation in degrees.
        text_alpha: Opacity of the text.

    Returns:
        matplotlib.text.Text: The stamped artist.
    """
    fig_w_in, _ = fig.get_size_inches()
    return fig.text(
        0.5,
        0.5,
        text,
        rotation=angle,
        ha="center",
        va="center",
        fontsize=fig_w_in * 6,
        fontweight="bold",
        color="white",
        alpha=text_alpha,
        zorder=1_000_000,
    )


class TestStampWatermark:
    """`stamp_watermark` places brand text sized as a fraction of the figure."""

    @pytest.mark.parametrize("frac", [0.2, 0.55, 0.9])
    def test_the_text_lands_at_the_requested_fraction(self, fig, frac):
        """The rendered text occupies `frac` of the figure's longer side.

        Args:
            fig: The figure fixture.
            frac: The requested fraction.

        Test scenario:
            This is the parameter's whole contract -- it is measured on what is
            actually rendered, not on a point size that happens to correlate.
        """
        brand, _ = stamp_watermark(fig, "earthlens", frac=frac)
        assert _share_of_figure(fig, brand) == pytest.approx(frac, abs=0.01), (
            f"asked for {frac}, rendered {_share_of_figure(fig, brand)}"
        )

    @pytest.mark.parametrize("text", ["eo", "earthlens", "a-much-longer-brand-name"])
    def test_the_fraction_is_independent_of_the_text_length(self, fig, text):
        """Any text lands at the same fraction.

        Args:
            fig: The figure fixture.
            text: The brand text under test.

        Test scenario:
            The defect this replaces: a point size scaled off the figure width
            renders a short word small and a long one off the canvas, because
            how much of the frame a string covers depends on its length. The
            notebook helper puts this same long name at ~128% of the figure.
        """
        brand, _ = stamp_watermark(fig, text, frac=0.55)
        assert _share_of_figure(fig, brand) == pytest.approx(0.55, abs=0.02), (
            f"{text!r} rendered at {_share_of_figure(fig, brand)}, not 0.55"
        )

    def test_it_beats_the_notebook_helper_on_a_long_name(self, fig):
        """The helper being replaced overflows where this one does not.

        Args:
            fig: The figure fixture.

        Test scenario:
            A positive assertion that the replacement is worth making: the same
            input that runs off the canvas with the old sizing rule stays on it
            here. Without this the length-independence test above could pass
            against a rule that was never broken.
        """
        long_name = "a-much-longer-brand-name"
        old = _notebook_helper(fig, long_name)
        new, _ = stamp_watermark(fig, long_name, frac=0.55)
        assert _share_of_figure(fig, old) > 1.0, (
            "precondition: the old rule should overflow the figure for this name"
        )
        assert _share_of_figure(fig, new) <= 1.0, (
            f"the new rule also overflowed: {_share_of_figure(fig, new)}"
        )

    def test_the_angle_reaches_the_artist(self, fig):
        """`angle` is applied as the text's rotation.

        Args:
            fig: The figure fixture.

        Test scenario:
            The diagonal is the point of the watermark; a dropped rotation
            would still render plausible-looking text.
        """
        brand, _ = stamp_watermark(fig, "earthlens", angle=45.0)
        assert brand.get_rotation() == pytest.approx(45.0), (
            f"rotation not applied: {brand.get_rotation()}"
        )

    def test_the_brand_text_is_translucent_and_unstroked(self, fig):
        """The large text carries alpha and no outline.

        Args:
            fig: The figure fixture.

        Test scenario:
            The design decision carried over from the notebooks: an outline on
            the big text makes it read as a solid caption rather than a
            watermark.
        """
        brand, _ = stamp_watermark(fig, "earthlens", alpha=0.4)
        assert brand.get_alpha() == pytest.approx(0.4), "alpha not applied"
        assert not brand.get_path_effects(), (
            "the brand text should carry no outline, only the credit line does"
        )

    def test_no_credit_line_by_default(self, fig):
        """Without `credit` only the brand text is drawn.

        Args:
            fig: The figure fixture.

        Test scenario:
            The credit is opt-in; returning `None` is what lets a caller tell
            the two cases apart without inspecting the figure.
        """
        brand, credit = stamp_watermark(fig, "earthlens")
        assert credit is None, f"expected no credit artist, got {credit!r}"
        assert brand in fig.texts, "the brand text was not added to the figure"


class TestStampWatermarkCredit:
    """The optional credit line along the bottom edge."""

    def test_the_credit_sits_at_the_margin(self, fig):
        """`margin` places the credit above the bottom edge.

        Args:
            fig: The figure fixture.

        Test scenario:
            The parameter replaces a hardcoded `0.014`, so it has to actually
            move the artist.
        """
        _, credit = stamp_watermark(fig, "earthlens", credit="example.org", margin=0.2)
        assert credit.get_position()[1] == pytest.approx(0.2), (
            f"margin not applied: {credit.get_position()}"
        )

    def test_the_credit_is_sized_by_fraction_too(self, fig):
        """`credit_frac` sets the credit's share of the figure.

        Args:
            fig: The figure fixture.

        Test scenario:
            The other hardcoded constant replaced -- a `7.5` point size that
            meant nothing in particular at any other figure size.
        """
        _, credit = stamp_watermark(
            fig,
            "earthlens",
            credit="github.com/serapeum-org/earthlens",
            credit_frac=0.4,
        )
        assert _share_of_figure(fig, credit) == pytest.approx(0.4, abs=0.02), (
            f"credit rendered at {_share_of_figure(fig, credit)}, not 0.4"
        )

    def test_the_credit_is_stroked(self, fig):
        """The credit line carries the outline the brand text does not.

        Args:
            fig: The figure fixture.

        Test scenario:
            The other half of the deliberate asymmetry: the credit is small
            enough that it needs a stroke to stay legible against arbitrary
            frame content.
        """
        brand, credit = stamp_watermark(fig, "earthlens", credit="example.org")
        assert credit.get_path_effects(), "the credit line should carry an outline"
        assert not brand.get_path_effects(), "the brand text should not"

    def test_the_credit_sits_above_the_brand_text(self, fig):
        """Z-order puts the credit on top.

        Args:
            fig: The figure fixture.

        Test scenario:
            The credit is the smallest element and the one that must never be
            occluded by the diagonal text it is stamped alongside.
        """
        brand, credit = stamp_watermark(fig, "earthlens", credit="example.org")
        assert credit.get_zorder() > brand.get_zorder(), (
            f"credit z={credit.get_zorder()} not above brand z={brand.get_zorder()}"
        )

    def test_the_credit_alpha_is_separate(self, fig):
        """`credit_alpha` is independent of the brand text's `alpha`.

        Args:
            fig: The figure fixture.

        Test scenario:
            The credit is information rather than decoration, so it defaults to
            opaque while the brand text is translucent.
        """
        brand, credit = stamp_watermark(
            fig, "earthlens", alpha=0.3, credit="example.org", credit_alpha=0.9
        )
        assert brand.get_alpha() == pytest.approx(0.3), "brand alpha wrong"
        assert credit.get_alpha() == pytest.approx(0.9), "credit alpha wrong"


class TestStampWatermarkValidation:
    """Invalid input is refused the way `stamp_mark` refuses it."""

    @pytest.mark.parametrize("bad", ["", "   ", 5])
    def test_a_blank_credit_raises(self, fig, bad):
        """A blank `credit` is refused rather than stamping an empty artist.

        Args:
            fig: The figure fixture.
            bad: The rejected value.

        Test scenario:
            A blank string looks like it means "no credit line" but would add
            an artist with nothing in it, which is the silently-accepted no-op
            the rest of this module refuses. `None` is absent from the table on
            purpose -- it is the documented way to ask for no credit, covered
            by `test_no_credit_line_by_default`.
        """
        with pytest.raises(ValueError, match="credit must be a non-empty string"):
            stamp_watermark(fig, "earthlens", credit=bad)

    @pytest.mark.parametrize("bad", ["", "   ", None, 5])
    def test_a_bad_text_raises(self, fig, bad):
        """`text` must be a non-empty string.

        Args:
            fig: The figure fixture.
            bad: The rejected value.

        Test scenario:
            An empty watermark renders nothing and reports nothing, which is
            the silent-no-op this package exists to avoid.
        """
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            stamp_watermark(fig, bad)

    @pytest.mark.parametrize("frac", [0.0, -0.1, 1.5])
    def test_an_out_of_range_frac_raises(self, fig, frac):
        """`frac` must be in (0, 1], as on `stamp_mark`.

        Args:
            fig: The figure fixture.
            frac: The rejected value.

        Test scenario:
            Matching `stamp_mark`'s bound exactly is the point -- two sibling
            functions that validate the same-named parameter differently is
            worse than neither validating.
        """
        with pytest.raises(ValueError, match=r"frac must be in \(0, 1\]"):
            stamp_watermark(fig, "earthlens", frac=frac)

    @pytest.mark.parametrize("alpha", [-0.1, 1.5])
    def test_an_out_of_range_alpha_raises(self, fig, alpha):
        """`alpha` must be in [0, 1].

        Args:
            fig: The figure fixture.
            alpha: The rejected value.

        Test scenario:
            Matplotlib silently clips an out-of-range alpha, so the caller
            would never learn the value they tuned was ignored.
        """
        with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
            stamp_watermark(fig, "earthlens", alpha=alpha)

    @pytest.mark.parametrize("angle", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_angle_raises(self, fig, angle):
        """`angle` must be a finite number of degrees.

        Args:
            fig: The figure fixture.
            angle: The rejected value.

        Test scenario:
            A NaN rotation renders the text at an undefined position rather
            than raising, which is the worst of both outcomes.
        """
        with pytest.raises(ValueError, match="angle must be a finite number"):
            stamp_watermark(fig, "earthlens", angle=angle)

    @pytest.mark.parametrize("credit_alpha", [-0.1, 1.5])
    def test_an_out_of_range_credit_alpha_raises(self, fig, credit_alpha):
        """`credit_alpha` must be in [0, 1].

        Args:
            fig: The figure fixture.
            credit_alpha: The rejected value.

        Test scenario:
            The credit's opacity is validated on the same terms as the brand
            text's.
        """
        with pytest.raises(ValueError, match=r"credit_alpha must be in \[0, 1\]"):
            stamp_watermark(
                fig, "earthlens", credit="example.org", credit_alpha=credit_alpha
            )

    @pytest.mark.parametrize("credit_frac", [0.0, 1.5])
    def test_an_out_of_range_credit_frac_raises(self, fig, credit_frac):
        """`credit_frac` must be in (0, 1].

        Args:
            fig: The figure fixture.
            credit_frac: The rejected value.

        Test scenario:
            Same bound as `frac`, for the same reason.
        """
        with pytest.raises(ValueError, match=r"credit_frac must be in \(0, 1\]"):
            stamp_watermark(
                fig, "earthlens", credit="example.org", credit_frac=credit_frac
            )

    @pytest.mark.parametrize("margin", [-0.1, 1.0, 1.5])
    def test_an_out_of_range_margin_raises(self, fig, margin):
        """`margin` must be in [0, 1), as on `stamp_mark`.

        Args:
            fig: The figure fixture.
            margin: The rejected value.

        Test scenario:
            `1.0` is in the table because the bound is half-open on that side,
            matching `stamp_mark` exactly.
        """
        with pytest.raises(ValueError, match=r"margin must be in \[0, 1\)"):
            stamp_watermark(fig, "earthlens", credit="example.org", margin=margin)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"credit_frac": 5.0},
            {"credit_alpha": 2.0},
            {"margin": 3.0},
        ],
    )
    def test_credit_only_parameters_are_unvalidated_without_a_credit(self, fig, kwargs):
        """The credit's parameters are only checked when a credit is drawn.

        Args:
            fig: The figure fixture.
            kwargs: Credit-only parameters with impossible values.

        Test scenario:
            Deliberate: they are dead when `credit is None`, so raising on them
            would reject a call that renders correctly. Pinned so the choice is
            visible rather than looking like an oversight.
        """
        brand, credit = stamp_watermark(fig, "earthlens", **kwargs)
        assert credit is None, "no credit should be drawn"
        assert brand in fig.texts, "the brand text should still be stamped"


class TestFitTextToFrac:
    """The sizing helper behind `stamp_watermark`'s `frac`."""

    def test_unmeasurable_text_is_left_alone(self, fig):
        """Text with no rendered extent does not divide by zero.

        Args:
            fig: The figure fixture.

        Test scenario:
            Only the empty string measures zero -- whitespace has real width
            and height, so `"   "` is resized like any other text.
            `stamp_watermark` refuses an empty `text` and an empty `credit`, so
            this guard is unreachable through the public function, which is
            exactly why it is worth exercising directly: without it the helper
            divides the target fraction by a zero extent.
        """
        artist = fig.text(0.5, 0.5, "", fontsize=20)
        _fit_text_to_frac(fig, artist, 0.5)
        assert artist.get_fontsize() == pytest.approx(20), (
            f"an unmeasurable artist should keep its size, got {artist.get_fontsize()}"
        )

    def test_it_converges_within_the_tolerance(self, fig):
        """The fit lands inside `_FIT_TOLERANCE` of the target.

        Args:
            fig: The figure fixture.

        Test scenario:
            Rendered size is near-linear in point size but hinting quantises
            glyphs, so the loop has to iterate. Pinning the tolerance is what
            says the loop converges rather than merely running.
        """
        artist = fig.text(0.5, 0.5, "earthlens", fontsize=8)
        _fit_text_to_frac(fig, artist, 0.6)
        assert _share_of_figure(fig, artist) == pytest.approx(0.6, abs=0.01), (
            f"fit landed at {_share_of_figure(fig, artist)}, not 0.6"
        )


class TestStampWatermarkWithStampMark:
    """The two helpers are designed to be used on the same figure."""

    def test_both_can_stamp_the_same_figure(self, fig, logo):
        """A corner logo and diagonal text coexist.

        Args:
            fig: The figure fixture.
            logo: The logo-array fixture.

        Test scenario:
            The use case from the earthlens notebooks that motivated this: a
            `stamp_mark` logo in the corner, `stamp_watermark` text across the
            middle, a credit line along the bottom.
        """
        mark_ax = stamp_mark(fig, logo, frac=0.18, corner="lower left")
        brand, credit = stamp_watermark(
            fig, "earthlens", credit="github.com/serapeum-org/earthlens"
        )

        assert mark_ax in fig.axes, "the mark axes is missing"
        assert brand in fig.texts and credit in fig.texts, "the text is missing"

    def test_the_mark_sits_above_the_brand_text(self, fig, logo):
        """Z-order puts the logo over the diagonal text.

        Args:
            fig: The figure fixture.
            logo: The logo-array fixture.

        Test scenario:
            A corner logo overlapping the diagonal text has to stay legible;
            the text is the background element of the two.
        """
        mark_ax = stamp_mark(fig, logo, frac=0.18, corner="lower left")
        brand, _ = stamp_watermark(fig, "earthlens")
        assert mark_ax.get_zorder() > brand.get_zorder(), (
            f"mark z={mark_ax.get_zorder()} not above text z={brand.get_zorder()}"
        )

    def test_the_figure_renders_with_both(self, fig, logo):
        """Drawing the figure with both stamps raises nothing.

        Args:
            fig: The figure fixture.
            logo: The logo-array fixture.

        Test scenario:
            The artists are excluded from layout, so a draw must not trip the
            tight-layout machinery the way an in-layout artist would.
        """
        stamp_mark(fig, logo, frac=0.18, corner="lower left")
        stamp_watermark(fig, "earthlens", credit="example.org")
        fig.canvas.draw()


def _rendered_array_glyph():
    """An `ArrayGlyph` that has been plotted.

    Returns:
        ArrayGlyph: A glyph holding a rendered figure.
    """
    glyph = ArrayGlyph(np.arange(60.0).reshape(6, 10))
    glyph.plot()
    return glyph


def _rendered_line_glyph():
    """A `LineGlyph` that has been rendered.

    Returns:
        LineGlyph: A glyph holding a rendered figure.
    """
    glyph = LineGlyph(np.arange(5.0), np.arange(5.0))
    glyph.line()
    return glyph


def _rendered_scatter_glyph():
    """A `ScatterGlyph` that has been rendered.

    Returns:
        ScatterGlyph: A glyph holding a rendered figure.
    """
    rng = np.random.default_rng(0)
    glyph = ScatterGlyph(rng.random(20), rng.random(20))
    glyph.plot()
    return glyph


def _rendered_kde_glyph():
    """A `KDEGlyph` that has been rendered.

    Returns:
        KDEGlyph: A glyph holding a rendered figure.
    """
    rng = np.random.default_rng(0)
    glyph = KDEGlyph(rng.random(60), rng.random(60))
    glyph.plot()
    return glyph


def _rendered_histogram_glyph():
    """A `HistogramGlyph` that has been rendered.

    Returns:
        HistogramGlyph: A glyph holding a rendered figure. Notable because it
        does not inherit `Glyph`.
    """
    glyph = HistogramGlyph(np.random.default_rng(0).normal(size=100))
    glyph.histogram()
    return glyph


def _rendered_globe_glyph():
    """A `TexturedGlobeGlyph` that has been drawn.

    Returns:
        TexturedGlobeGlyph: A glyph holding a rendered figure. Notable because
        it does not inherit `Glyph` either.
    """
    texture = np.zeros((16, 32, 4))
    texture[..., 3] = 1.0
    glyph = TexturedGlobeGlyph(texture)
    glyph.draw()
    return glyph


#: Every glyph shape the mixin has to work on: the `Glyph` subclasses, plus the
#: two classes that do not inherit `Glyph` and keep their figure elsewhere.
GLYPH_BUILDERS = {
    "array": _rendered_array_glyph,
    "line": _rendered_line_glyph,
    "scatter": _rendered_scatter_glyph,
    "kde": _rendered_kde_glyph,
    "histogram": _rendered_histogram_glyph,
    "globe": _rendered_globe_glyph,
}


class TestWatermarkMixin:
    """Both stamps are reachable as glyph methods, on every glyph."""

    @pytest.mark.parametrize("name", sorted(GLYPH_BUILDERS))
    def test_stamp_watermark_is_a_method(self, name, logo):
        """`glyph.stamp_watermark(...)` stamps the glyph's own figure.

        Args:
            name: The glyph kind under test.
            logo: The logo-array fixture (unused here, keeps the signature
                uniform with its sibling).

        Test scenario:
            The point of the mixin: no import, no passing `glyph.fig` by hand.
            Every glyph is covered because two of them do not inherit `Glyph`
            and would otherwise be missed.
        """
        glyph = GLYPH_BUILDERS[name]()
        brand, credit = glyph.stamp_watermark("cleopatra")

        assert brand.get_text() == "cleopatra", f"{name}: text not stamped"
        assert credit is None, f"{name}: unexpected credit line"
        assert brand in glyph._watermark_figure().texts, (
            f"{name}: the artist did not land on the glyph's own figure"
        )
        plt.close("all")

    @pytest.mark.parametrize("name", sorted(GLYPH_BUILDERS))
    def test_stamp_mark_is_a_method(self, name, logo):
        """`glyph.stamp_mark(...)` stamps the glyph's own figure.

        Args:
            name: The glyph kind under test.
            logo: The logo-array fixture.

        Test scenario:
            `stamp_mark` is wired at the same time as `stamp_watermark` on
            purpose -- giving the newer function sugar the older one lacked
            would be a worse inconsistency than neither having it.
        """
        glyph = GLYPH_BUILDERS[name]()
        ax = glyph.stamp_mark(logo, frac=0.2)

        assert ax in glyph._watermark_figure().axes, (
            f"{name}: the mark did not land on the glyph's own figure"
        )
        plt.close("all")

    def test_keyword_arguments_reach_the_function(self):
        """The methods forward their keywords verbatim.

        Test scenario:
            They are sugar, not a reduced surface; a caller must be able to
            reach every parameter of the free function through them.
        """
        glyph = _rendered_array_glyph()
        brand, credit = glyph.stamp_watermark(
            "cleopatra", angle=12.0, alpha=0.3, credit="example.org", margin=0.2
        )
        assert brand.get_rotation() == pytest.approx(12.0), "angle not forwarded"
        assert brand.get_alpha() == pytest.approx(0.3), "alpha not forwarded"
        assert credit.get_position()[1] == pytest.approx(0.2), "margin not forwarded"
        plt.close("all")

    def test_validation_still_applies_through_the_method(self):
        """Invalid input raises the same way as through the free function.

        Test scenario:
            The sugar must not become a hole in the validation that the free
            function does.
        """
        glyph = _rendered_array_glyph()
        with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
            glyph.stamp_watermark("cleopatra", alpha=2.0)
        plt.close("all")

    def test_an_unrendered_glyph_raises_a_useful_error(self):
        """Stamping before rendering says so, rather than failing obscurely.

        Test scenario:
            A glyph that has not been plotted has no figure. The message has to
            name the glyph and the fix, since `AttributeError: 'NoneType'` from
            somewhere inside matplotlib would not.
        """
        glyph = ArrayGlyph(np.arange(6.0).reshape(2, 3))
        with pytest.raises(ValueError, match="has no figure to stamp yet"):
            glyph.stamp_watermark("cleopatra")

    def test_the_free_functions_are_unchanged(self, fig, logo):
        """Wiring the methods did not take the free functions away.

        Test scenario:
            They remain the primitives -- the mixin only spares the import, and
            a figure the glyph does not own still has to go through them.
        """
        ax = stamp_mark(fig, logo, frac=0.2)
        brand, _ = stamp_watermark(fig, "cleopatra")
        assert ax in fig.axes and brand in fig.texts, (
            "the free functions should still stamp an arbitrary figure"
        )


class _DetachedAxes:
    """An axes-like object whose figure is gone.

    Matplotlib does not normally produce one, which is why the guard against it
    needs a stand-in to be exercised at all.
    """

    def get_figure(self):
        """Report no figure.

        Returns:
            None: Always.
        """
        return None


class _GlyphWithDetachedAxes(WatermarkMixin):
    """A glyph-like holder whose only axes cannot name a figure."""

    def __init__(self):
        """Hold a detached axes and nothing else."""
        self._ax = _DetachedAxes()


class TestWatermarkFigureResolution:
    """`_watermark_figure` refuses rather than returning something unusable."""

    def test_an_axes_without_a_figure_is_not_accepted(self):
        """A detached axes falls through to the error, not to `None`.

        Test scenario:
            Returning the `None` an axes reported would push the failure into
            matplotlib, where it surfaces as an `AttributeError` on `NoneType`
            far from the cause. The resolver keeps the explanation instead.
        """
        with pytest.raises(ValueError, match="has no figure to stamp yet"):
            _GlyphWithDetachedAxes().stamp_watermark("cleopatra")


class TestRenderedFigureTracking:
    """`_rendered_fig` records where a render landed, without disturbing `_fig`."""

    def test_histogram_glyph_tracks_the_figure_it_drew_on(self):
        """A rendered `HistogramGlyph` can resolve a figure.

        Test scenario:
            It does not inherit `Glyph` and does not keep `fig`, so without the
            tracking the inherited methods would exist and always raise -- the
            advertised-but-dead method this package keeps having to fix.
        """
        glyph = HistogramGlyph(np.random.default_rng(0).normal(size=100))
        assert glyph._rendered_fig is None, "precondition: nothing rendered yet"
        figure, _, _ = glyph.histogram()
        assert glyph._watermark_figure() is figure, (
            "the tracked figure is not the one histogram() drew on"
        )
        plt.close("all")

    @pytest.mark.parametrize("method", ["boxplot", "multiboxplot", "stripes"])
    def test_every_histogram_render_path_tracks(self, method):
        """The other three render methods track too.

        Args:
            method: The render method under test.

        Test scenario:
            All four funnel through two resolvers; recording there rather than
            in each method is what makes this hold without four separate edits
            to keep in step.
        """
        values = (
            np.random.default_rng(0).normal(size=(100, 3))
            if method == "multiboxplot"
            else np.random.default_rng(0).normal(size=100)
        )
        glyph = HistogramGlyph(values)
        getattr(glyph, method)()
        assert glyph._rendered_fig is not None, f"{method} did not track its figure"
        plt.close("all")

    def test_a_glyph_holding_only_an_axes_resolves_through_it(self):
        """An axes-bound glyph can be stamped before it renders.

        Test scenario:
            `HistogramGlyph(values, ax=...)` has no figure of its own until it
            renders -- `_fig` stays unset and nothing has been drawn -- but the
            axes it was handed knows its figure, so the stamp can still land
            somewhere sensible rather than refusing.
        """
        figure, axes = plt.subplots()
        glyph = HistogramGlyph(np.random.default_rng(0).normal(size=50), ax=axes)
        assert glyph._rendered_fig is None and glyph._fig is None, (
            "precondition: the glyph holds only an axes"
        )
        brand, _ = glyph.stamp_watermark("cleopatra")
        assert brand in figure.texts, "the stamp did not land on the axes' figure"
        plt.close("all")

    def test_the_construction_figure_keeps_its_meaning(self):
        """Tracking does not turn `_fig` into "the last figure drawn on".

        Test scenario:
            `_fig` is the construction-time target, consulted on every render to
            decide where to draw. Had the tracking reused it, a glyph built
            without a figure would acquire one after the first render and the
            second render would land on it instead of a fresh figure -- a real
            behaviour change hiding behind a convenience feature.
        """
        glyph = HistogramGlyph(np.random.default_rng(0).normal(size=100))
        first, _, _ = glyph.histogram()
        assert glyph._fig is None, "_fig should still be unset after a render"
        second, _, _ = glyph.histogram()
        assert second is not first, (
            "a second render should make its own figure, not reuse the first"
        )
        plt.close("all")

    def test_a_construction_figure_is_still_honoured(self):
        """A figure passed at construction is still the render target.

        Test scenario:
            The other half of the same guarantee: the tracking must not stop
            `_fig` being used when it *was* supplied.
        """
        supplied = plt.figure(figsize=(6.0, 4.0))
        glyph = HistogramGlyph(np.random.default_rng(0).normal(size=100), fig=supplied)
        drawn, _, _ = glyph.histogram()
        assert drawn is supplied, "the construction figure was not used"
        assert glyph._watermark_figure() is supplied, "tracking disagrees"
        plt.close("all")
