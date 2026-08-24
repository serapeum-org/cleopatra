"""Tests for the figure watermark / brand-mark helper -- issue #312.

Covers `cleopatra.styling.watermark.stamp_mark`: corner placement in
figure-fraction coordinates, dpi-invariant sizing, undistorted aspect, the
optional gaussian-blurred drop shadow, image-input handling (RGBA/RGB arrays,
float arrays, file paths), and input validation.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from cleopatra.styling.watermark import _CORNERS, stamp_mark


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
        ax = stamp_mark(fig, logo, frac=frac, corner=corner, margin=margin, shadow=False)
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
        assert np.allclose(at_100, at_300), f"position changed with dpi: {at_100} vs {at_300}"

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
        assert np.isclose(h_in / w_in, 40 / 80), f"aspect distorted: {h_in / w_in} != {40 / 80}"

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
        assert np.isclose((h * 6.0) / (w * 8.0), 400 / 40), f"tall logo distorted: {(w, h)}"

    def test_shadow_adds_a_second_axes(self, fig, logo):
        """`shadow=True` draws the shadow on its own axes beneath the mark.

        Test scenario:
            Stamping with a shadow adds two axes (shadow + mark), and the
            shadow's zorder is below the mark's.
        """
        n_before = len(fig.axes)
        ax = stamp_mark(fig, logo, shadow=True)
        assert len(fig.axes) - n_before == 2, "shadow should add a mark axes and a shadow axes"
        others = [a for a in fig.axes if a is not ax and a.get_zorder() >= 1_000_000]
        assert others, "no shadow axes found"
        assert others[0].get_zorder() < ax.get_zorder(), "shadow must sit beneath the mark"

    def test_no_shadow_adds_single_axes(self, fig, logo):
        """`shadow=False` draws only the mark axes.

        Test scenario:
            Without a shadow exactly one axes is added.
        """
        n_before = len(fig.axes)
        stamp_mark(fig, logo, shadow=False)
        assert len(fig.axes) - n_before == 1, "no-shadow stamp should add exactly one axes"

    def test_returns_frameless_axes_above_content(self, fig, logo):
        """The mark axes is frameless, off, and above ordinary content.

        Test scenario:
            The returned axes has its frame off (no white box), no visible
            axis, and a very high zorder so it draws on top of the plot.
        """
        ax = stamp_mark(fig, logo, shadow=False)
        assert not ax.get_frame_on(), "mark axes must be frameless (no background box)"
        assert not ax.axison, "mark axes must have its axis turned off"
        assert ax.get_zorder() >= 1_000_000, "mark must sit above ordinary figure content"

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
            ax_path.get_position().bounds, (0.865, 0.025, 0.11, 0.11 * 0.5 * (8.0 / 6.0))
        ), f"file-path placement wrong: {ax_path.get_position().bounds}"

    def test_frac_controls_width(self, fig, logo):
        """A larger `frac` yields a proportionally wider mark.

        Test scenario:
            Doubling `frac` doubles the mark's figure-fraction width.
        """
        small = stamp_mark(fig, logo, frac=0.1, shadow=False).get_position().width
        big = stamp_mark(fig, logo, frac=0.2, shadow=False).get_position().width
        assert np.isclose(big, 2 * small), f"frac should scale width linearly: {big} vs {small}"

    def test_corners_constant_exposes_four_anchors(self):
        """`_CORNERS` lists exactly the four documented anchors.

        Test scenario:
            The accepted-corner set matches the documented four.
        """
        assert set(_CORNERS) == {"lower right", "lower left", "upper right", "upper left"}
