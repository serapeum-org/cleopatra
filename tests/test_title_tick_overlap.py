"""Tests that a multi-line title clears the top x tick labels -- issue #345.

`ArrayGlyph` renders through `ax.matshow`, which puts the x tick labels on the
top spine. Matplotlib raises the title above them, but anchors the text's *first*
line, so every further line is drawn back down through the labels it just
cleared. A single-line title therefore cleared them and a two-line one did not.

These assert the title's bounding box does not intersect the tick labels' or the
axis offset text, at an explicit figsize -- the auto-computed one leaves a tall
enough axes to mask the bug, so a test without `figsize` would not catch it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cleopatra.glyphs.base.glyph import multiline_title_pad
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph

#: A UTM window, so the y axis carries a `1e6` offset text like a real granule.
EXTENT = [530000.0, 4124000.0, 542000.0, 4130000.0]

TWO_LINE = (
    "OPERA RTC-S1 VH backscatter\n"
    "OPERA_L2_RTC-S1_T042-088920-IW3_20240605T140837Z_S1A_30_v1.0_VH.tif"
)


@pytest.fixture
def arr():
    """Provide a 200x400 array standing in for a satellite granule.

    Returns:
        np.ndarray: The backscatter-like array to render.
    """
    return np.random.default_rng(0).normal(size=(200, 400)).astype("float32")


def _collisions(arr, title, figsize=(8, 6), title_size=15):
    """Render and report what the title overlaps.

    Args:
        arr: The array to plot.
        title: The title text.
        figsize: Explicit figure size; the bug needs one.
        title_size: The title font size.

    Returns:
        tuple: The overlapping tick label texts, and whether the y-axis offset
        text is overlapped.
    """
    options = {"title": title, "extent": list(EXTENT), "title_size": title_size}
    if figsize is not None:
        # Passing `figsize=None` would still count as explicit and switch the
        # auto-sizing off, which is the very path the `figsize=None` case exists
        # to exercise.
        options["figsize"] = figsize
    glyph = ArrayGlyph(arr, **options)
    fig, ax = glyph.plot(cmap="gray")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = ax.title.get_window_extent(renderer)

    labels = [
        artist.get_text()
        for tick in ax.xaxis.majorTicks
        for artist in (tick.label1, tick.label2)
        if artist.get_visible()
        and artist.get_text()
        and box.overlaps(artist.get_window_extent(renderer))
    ]
    offset = ax.yaxis.get_offset_text()
    offset_hit = bool(
        offset.get_visible()
        and offset.get_text()
        and box.overlaps(offset.get_window_extent(renderer))
    )
    plt.close(fig)
    return labels, offset_hit


class TestMultiLineTitleClearsTickLabels:
    """The title must not be drawn through the top tick labels."""

    @pytest.mark.parametrize(
        "figsize", [(8, 6), (6, 4), (5, 3), (4, 2.5), (12, 3), (3, 6)]
    )
    def test_two_line_title_clears_the_tick_band(self, arr, figsize):
        """A two-line title clears the labels at every figure size.

        Args:
            arr: The array fixture.
            figsize: The explicit figure size under test.

        Test scenario:
            The reported case. Every one of these sizes collided before; the
            auto-computed size did not, which is why the bug went unnoticed.
        """
        labels, _ = _collisions(arr, TWO_LINE, figsize=figsize)
        assert not labels, f"figsize={figsize}: title overlaps tick labels {labels}"

    @pytest.mark.parametrize("lines", [1, 2, 3, 5])
    def test_any_line_count_clears(self, arr, lines):
        """One line through five all clear the labels.

        Args:
            arr: The array fixture.
            lines: How many title lines to render.

        Test scenario:
            A single-line title always cleared them; the pad must not break that
            while fixing the taller ones.
        """
        title = "\n".join(f"line {i}" for i in range(lines))
        labels, _ = _collisions(arr, title)
        assert not labels, f"{lines}-line title overlaps {labels}"

    def test_enlarged_title_clears_the_offset_text(self, arr):
        """A large two-line title clears the `1e6` y-axis offset too.

        Test scenario:
            The offset sits in the same band as the tick labels and was struck
            once the title was enlarged, even though it was clear at the default
            size.
        """
        labels, offset_hit = _collisions(arr, TWO_LINE, title_size=26)
        assert not labels, f"title overlaps tick labels {labels}"
        assert not offset_hit, "title overlaps the y-axis offset text"

    def test_auto_figsize_still_clears(self, arr):
        """The auto-computed figure size stays clear.

        Test scenario:
            It was already clear, so the pad must not push the title into
            anything else. `figsize` is omitted rather than passed as `None`,
            since passing it at all marks it explicit and turns the auto-sizing
            off -- which would quietly test the wrong path.
        """
        glyph = ArrayGlyph(arr, title=TWO_LINE, extent=list(EXTENT))
        fig, ax = glyph.plot(cmap="gray")
        assert fig.get_size_inches().tolist() != [8.0, 6.0], (
            "precondition: the figure was auto-sized, not left at the default"
        )
        plt.close(fig)

        labels, offset_hit = _collisions(arr, TWO_LINE, figsize=None)
        assert not labels and not offset_hit


class TestMultilineTitlePad:
    """Direct tests for the pad calculation."""

    def test_single_line_gets_no_pad(self):
        """A one-line title keeps matplotlib's default pad.

        Test scenario:
            `None` means "leave the default", so a single-line title renders
            exactly as before.
        """
        fig, ax = plt.subplots()
        ax.matshow(np.zeros((4, 4)))
        assert multiline_title_pad(ax, "one line", 15) is None
        plt.close(fig)

    def test_no_pad_when_labels_are_not_on_top(self):
        """With bottom tick labels the default pad is already right.

        Test scenario:
            Matplotlib's own raise only under-shoots when labels sit on the top
            spine, so nothing is added otherwise.
        """
        fig, ax = plt.subplots()
        ax.imshow(np.zeros((4, 4)))  # imshow leaves the labels on the bottom
        assert multiline_title_pad(ax, "two\nlines", 15) is None
        plt.close(fig)

    @pytest.mark.parametrize("lines, size", [(2, 15), (3, 15), (2, 26), (5, 10)])
    def test_pad_grows_with_lines_and_font_size(self, lines, size):
        """The pad adds one line height per line after the first.

        Args:
            lines: The title's line count.
            size: The title font size.

        Test scenario:
            The deficit is exactly the height of the lines that hang below the
            anchor, so the pad is that height plus the default.
        """
        fig, ax = plt.subplots()
        ax.matshow(np.zeros((4, 4)))
        one_line = multiline_title_pad(ax, "x", size) or plt.rcParams["axes.titlepad"]
        title = "\n".join("x" for _ in range(lines))
        pad = multiline_title_pad(ax, title, size)

        # Assert the property rather than the formula: the pad must cover the
        # rendered height of the lines that hang below the anchor. Restating
        # `(lines - 1) * size * 1.2` here could not catch a wrong formula.
        # Measured on the axes' own title after a real draw, so the assertion
        # uses the same `Text` -- same font, same line spacing -- that the pad
        # was computed for, rather than a stand-in that could differ.
        ax.set_title(title, fontsize=size)
        fig.canvas.draw()
        height_px = ax.title.get_window_extent(fig.canvas.get_renderer()).height
        # The pad is in points and the rendered height in pixels, so one has to
        # be converted before they can be compared at all.
        height_points = height_px * 72.0 / fig.dpi
        needed = height_points * (lines - 1) / lines
        assert pad - one_line == pytest.approx(needed, rel=1e-3), (
            f"pad {pad} adds {pad - one_line:.2f}pt for {lines} lines, against "
            f"the {needed:.2f}pt those extra lines actually occupy"
        )
        plt.close(fig)
