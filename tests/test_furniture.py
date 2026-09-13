"""Tests for the map/chart furniture helpers -- issue #352.

Covers `cleopatra.styling.furniture.add_scale_bar` and `add_north_arrow`:
corner placement in axes-fraction coordinates, the segmented bar geometry,
tick numbers and caption placement, the backing box, zorder above the data,
stability across a limits change, the north-arrow styles and rotation, input
validation, and the `GeoMixin` sugar methods.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.patches import Polygon, Rectangle

from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
from cleopatra.styling.furniture import (
    _NORTH_STYLES,
    NorthArrow,
    ScaleBar,
    _north_arrow_patches,
    _resolve_box,
    _rose_base,
    _scale_bar_ticks,
    _segment_fills,
    add_north_arrow,
    add_scale_bar,
)


@pytest.fixture
def ax():
    """Provide a populated axes spanning 0..500000 in both directions.

    Returns:
        matplotlib.axes.Axes: An axes carrying an imshow so furniture is drawn
        over real content, with a known 500000-unit x-range.
    """
    figure, axes = plt.subplots(figsize=(8.0, 6.0))
    axes.imshow(np.arange(100).reshape(10, 10))
    # imshow forces an equal aspect (its drawn box shrinks); "auto" lets the
    # axes fill its subplot box so a transAxes inset's figure-fraction position
    # maps cleanly back to axes fraction in the placement assertions.
    axes.set_aspect("auto")
    axes.set_xlim(0.0, 500_000.0)
    axes.set_ylim(0.0, 500_000.0)
    yield axes
    plt.close(figure)


def _axfrac(parent, inset):
    """Return an inset axes' position as (x0, y0, w, h) in parent-axes fraction.

    `Axes.get_position` reports figure-fraction bounds; a `transAxes` inset is
    placed relative to the parent, so convert back through the parent's own box.

    Args:
        parent: The parent axes the inset was drawn on.
        inset: The inset axes returned by a furniture function.

    Returns:
        tuple[float, float, float, float]: The inset origin and size as
        fractions of the parent axes.
    """
    p = parent.get_position()
    i = inset.get_position()
    return (
        (i.x0 - p.x0) / p.width,
        (i.y0 - p.y0) / p.height,
        i.width / p.width,
        i.height / p.height,
    )


class TestSegmentFills:
    """Tests for the `_segment_fills` helper."""

    def test_alternating(self):
        """Blocks alternate between `color` and `edge_color`.

        Test scenario:
            Four blocks give color, edge, color, edge.
        """
        assert _segment_fills(4, "black", "white") == [
            "black",
            "white",
            "black",
            "white",
        ], "blocks should alternate"

    def test_single_block(self):
        """A single block is the primary colour.

        Test scenario:
            `segments=1` yields one `color` fill.
        """
        assert _segment_fills(1, "black", "white") == ["black"], "one block is color"


class TestResolveBox:
    """Tests for the `_resolve_box` helper."""

    @pytest.mark.parametrize("falsy", [None, False])
    def test_falsy_no_box(self, falsy):
        """A falsy `box` draws no panel.

        Args:
            falsy: `None` or `False`.

        Test scenario:
            `_resolve_box` returns `None`.
        """
        assert _resolve_box(falsy) is None, "no panel for a falsy box"

    def test_true_default_panel(self):
        """`True` gives a translucent white panel.

        Test scenario:
            The default facecolor is white.
        """
        kw = _resolve_box(True)
        assert kw["facecolor"] == "white", "True should be a white panel"

    def test_string_facecolor(self):
        """A string sets the panel face colour.

        Test scenario:
            `box="red"` sets facecolor red.
        """
        assert _resolve_box("red")["facecolor"] == "red", "string sets facecolor"

    def test_dict_merges_over_defaults(self):
        """A dict merges over the defaults.

        Test scenario:
            An explicit edgecolor overrides while the white face default stays.
        """
        kw = _resolve_box({"edgecolor": "navy"})
        assert kw["edgecolor"] == "navy", "explicit edgecolor should override"
        assert kw["facecolor"] == "white", "the white face default should remain"


class TestScaleBarTicksHelper:
    """Tests for the `_scale_bar_ticks` helper (explicit / falsy paths)."""

    def test_true_defers_to_caller(self):
        """`True` returns empty lists (boundaries added from `segments`).

        Test scenario:
            The helper does not know `segments`, so `True` yields no ticks here.
        """
        assert _scale_bar_ticks(True, 100.0) == ([], []), "True is handled upstream"

    def test_false_no_ticks(self):
        """`False` yields no ticks.

        Test scenario:
            Empty position and value lists.
        """
        assert _scale_bar_ticks(False, 100.0) == ([], []), "False -> no ticks"

    def test_explicit_positions(self):
        """An explicit sequence becomes fractions and values.

        Test scenario:
            Positions 0/50/100 on a length-100 bar map to fractions 0/0.5/1.
        """
        fracs, vals = _scale_bar_ticks([0.0, 50.0, 100.0], 100.0)
        assert fracs == [0.0, 0.5, 1.0], f"unexpected fractions {fracs}"
        assert vals == [0.0, 50.0, 100.0], f"unexpected values {vals}"

    def test_out_of_range_raises(self):
        """A tick outside `[0, length]` raises.

        Test scenario:
            A position beyond the bar length is rejected.
        """
        with pytest.raises(ValueError, match="outside the bar range"):
            _scale_bar_ticks([0.0, 150.0], 100.0)


class TestAddScaleBar:
    """Tests for `add_scale_bar`."""

    @pytest.mark.parametrize(
        "location, at_right, at_top",
        [
            ("lower left", False, False),
            ("lower right", True, False),
            ("upper left", False, True),
            ("upper right", True, True),
        ],
    )
    def test_corner_placement(self, ax, location, at_right, at_top):
        """The bar anchors in each of the four corners.

        Args:
            ax: The populated axes fixture.
            location: The corner under test.
            at_right: Whether the corner is on the right.
            at_top: Whether the corner is on the top.

        Test scenario:
            The inset's axes-fraction origin sits `pad` from the expected edges;
            a 100000-unit bar on a 500000 range is 0.2 wide.
        """
        bar = add_scale_bar(ax, 100_000, ScaleBar(location=location, pad=0.025))
        x0, y0, w, h = _axfrac(ax, bar)
        assert w == pytest.approx(0.2, abs=1e-6), f"bar width {w} != 0.2"
        assert x0 == pytest.approx(0.775 if at_right else 0.025, abs=1e-6), f"x0={x0}"
        expected_y = (1.0 - 0.025 - h) if at_top else 0.025
        assert y0 == pytest.approx(expected_y, abs=1e-6), f"y0={y0}"

    def test_returns_inset_axes(self, ax):
        """The function returns the inset axes the bar is on.

        Test scenario:
            An `Axes` is returned and it is not the parent.
        """
        bar = add_scale_bar(ax, 100_000)
        assert isinstance(bar, Axes), "should return an Axes"
        assert bar is not ax, "should return the inset axes, not the parent"

    def test_segment_geometry(self, ax):
        """`segments` alternating blocks tile the bar left to right.

        Test scenario:
            Four blocks each span a quarter of the inset width.
        """
        bar = add_scale_bar(ax, 100_000, ScaleBar(segments=4))
        assert len(bar.patches) == 4, "four blocks"
        first = bar.patches[0]
        assert first.get_xy() == (0.0, 0.0), "first block starts at the origin"
        assert first.get_width() == pytest.approx(0.25), "each block is a quarter wide"

    def test_segment_fill_alternation(self, ax):
        """The blocks alternate fill colours.

        Test scenario:
            Block 0 is `color` and block 1 is `edge_color`.
        """
        bar = add_scale_bar(
            ax, 100_000, ScaleBar(segments=2, color="black", edge_color="white")
        )
        assert bar.patches[0].get_facecolor() != bar.patches[1].get_facecolor(), (
            "adjacent blocks differ"
        )

    def test_boundary_tick_numbers(self, ax):
        """`ticks=True` numbers the segment boundaries `0 .. length`.

        Test scenario:
            A 4-segment 100000 bar labels 0, 25000, 50000, 75000, 100000.
        """
        add_scale_bar(ax, 100_000, ScaleBar(segments=4, ticks=True, label="d"))
        texts = [t.get_text() for t in ax.texts]
        for expected in ("0", "25000", "50000", "75000", "100000"):
            assert expected in texts, f"missing tick number {expected} in {texts}"

    def test_explicit_ticks(self, ax):
        """An explicit tick sequence numbers those data positions.

        Test scenario:
            Ticks at 0 / 250000 (as data values, not boundaries).
        """
        add_scale_bar(ax, 400_000, ScaleBar(ticks=[0.0, 250_000.0], label="d"))
        texts = [t.get_text() for t in ax.texts]
        assert "250000" in texts, f"explicit tick 250000 missing in {texts}"

    def test_ticks_false_only_caption(self, ax):
        """`ticks=False` draws only the caption, no boundary numbers.

        Test scenario:
            The only parent text is the caption.
        """
        add_scale_bar(ax, 100_000, ScaleBar(ticks=False, label="100 km"))
        assert [t.get_text() for t in ax.texts] == ["100 km"], "only the caption"

    def test_default_label_is_length(self, ax):
        """The caption defaults to the length formatted with `:g`.

        Test scenario:
            No `label` gives a `"100000"` caption.
        """
        add_scale_bar(ax, 100_000, ScaleBar(ticks=False))
        assert ax.texts[-1].get_text() == "100000", "caption defaults to the length"

    def test_caption_centered(self, ax):
        """The caption is centred under the bar.

        Test scenario:
            The caption x is the bar centre (0.025 + 0.2/2 = 0.125).
        """
        add_scale_bar(
            ax, 100_000, ScaleBar(location="lower left", ticks=False, label="100 km")
        )
        caption = ax.texts[-1]
        assert caption.get_position()[0] == pytest.approx(0.125), "caption centred"
        assert caption.get_ha() == "center", "caption horizontally centred"

    @pytest.mark.parametrize("side, va", [("bottom", "top"), ("top", "bottom")])
    def test_label_location_side(self, ax, side, va):
        """`label_location` puts the caption below or above the bar.

        Args:
            ax: The axes fixture.
            side: The `label_location` value.
            va: The expected vertical alignment.

        Test scenario:
            A bottom caption is top-aligned (grows down); a top one is
            bottom-aligned (grows up).
        """
        add_scale_bar(
            ax, 100_000, ScaleBar(ticks=False, label="x", label_location=side)
        )
        assert ax.texts[-1].get_va() == va, f"{side} caption should be va={va}"

    def test_default_text_stays_on_axes(self, ax):
        """The default call keeps every tick number and caption within the axes.

        Test scenario:
            `add_scale_bar(ax, length)` — lower-right corner with the auto label
            side — draws all its text at axes-fraction `y` inside `[0, 1]`, not
            spilling off the bottom edge over the axis tick labels.
        """
        add_scale_bar(ax, 100_000)
        ys = [t.get_position()[1] for t in ax.texts]
        assert ys, "the default call should draw tick numbers and a caption"
        assert all(0.0 <= y <= 1.0 for y in ys), (
            f"furniture text spilled off the axes: {ys}"
        )

    @pytest.mark.parametrize(
        "location, va",
        [
            ("lower left", "bottom"),
            ("lower right", "bottom"),
            ("upper left", "top"),
            ("upper right", "top"),
        ],
    )
    def test_auto_label_side_faces_interior(self, ax, location, va):
        """The auto label side faces the axes interior for each corner.

        Args:
            ax: The axes fixture.
            location: The corner under test.
            va: The expected caption vertical alignment (lower corners label
                above the bar -> va="bottom"; upper corners below -> va="top").

        Test scenario:
            With no explicit `label_location`, a lower corner labels above the
            bar and an upper corner below, so the caption stays interior.
        """
        add_scale_bar(ax, 100_000, ScaleBar(location=location, ticks=False, label="x"))
        assert ax.texts[-1].get_va() == va, f"{location} caption should be va={va}"

    def test_box_draws_panel(self, ax):
        """`box` adds one backing rectangle on the parent axes.

        Test scenario:
            The parent gains a single `Rectangle` patch.
        """
        before = len(ax.patches)
        add_scale_bar(ax, 100_000, ScaleBar(box=True))
        rects = [p for p in ax.patches if isinstance(p, Rectangle)]
        assert len(ax.patches) == before + 1, "one backing panel added"
        assert rects, "the backing panel is a Rectangle"

    def test_box_top_label(self, ax):
        """A backing box also works with a top caption.

        Test scenario:
            `label_location="top"` with `box=True` still draws one panel (the
            box grows above the bar rather than below).
        """
        before = len(ax.patches)
        add_scale_bar(ax, 100_000, ScaleBar(box=True, label_location="top"))
        assert len(ax.patches) == before + 1, "one backing panel for a top caption"

    def test_box_bottom_label(self, ax):
        """A backing box also works with a bottom caption.

        Test scenario:
            `label_location="bottom"` with `box=True` draws one panel that grows
            below the bar (the `sign < 0` branch).
        """
        before = len(ax.patches)
        add_scale_bar(ax, 100_000, ScaleBar(box=True, label_location="bottom"))
        assert len(ax.patches) == before + 1, "one backing panel for a bottom caption"

    def test_box_covers_caption_without_ticks(self, ax):
        """The backing box reserves room for the always-drawn caption.

        Test scenario:
            With `ticks=False` and no explicit label (the caption still defaults
            to the length), the panel's vertical span contains the caption's
            anchor, rather than sizing itself as if no text were drawn.
        """
        bar = add_scale_bar(ax, 100_000, ScaleBar(box=True, ticks=False))
        panel = ax.patches[-1]
        py0 = panel.get_y()
        py1 = py0 + panel.get_height()
        caption_y = ax.texts[-1].get_position()[1]
        assert py0 <= caption_y <= py1, (
            f"caption y={caption_y} not inside box [{py0}, {py1}]"
        )
        assert bar is not ax

    def test_box_color(self, ax):
        """A string `box` sets the panel face colour.

        Test scenario:
            `box="yellow"` draws a yellow panel.
        """
        add_scale_bar(ax, 100_000, ScaleBar(box="yellow"))
        panel = ax.patches[-1]
        assert panel.get_facecolor()[:3] == pytest.approx((1.0, 1.0, 0.0)), (
            "panel should be yellow"
        )

    def test_zorder_above_data(self, ax):
        """The furniture sits above the data.

        Test scenario:
            The inset zorder exceeds the image zorder.
        """
        img_z = ax.get_images()[0].get_zorder()
        bar = add_scale_bar(ax, 100_000)
        assert bar.get_zorder() > img_z, "furniture should be above the data"

    def test_custom_zorder(self, ax):
        """An explicit `zorder` is honoured.

        Test scenario:
            `zorder=42` sets the inset draw order near that value.
        """
        bar = add_scale_bar(ax, 100_000, ScaleBar(zorder=42.0))
        assert bar.get_zorder() >= 42.0, "explicit zorder should be honoured"

    def test_stable_across_set_xlim(self, ax):
        """The bar stays anchored across a limits change.

        Test scenario:
            Changing the x-limits does not move the inset (axes-fraction
            placement), unlike a data-coordinate rectangle.
        """
        bar = add_scale_bar(ax, 100_000, ScaleBar(location="lower left"))
        before = tuple(bar.get_position().bounds)
        ax.set_xlim(0.0, 1_000_000.0)
        assert tuple(bar.get_position().bounds) == before, "placement must be stable"

    def test_bad_location_raises(self, ax):
        """An unknown `location` raises.

        Test scenario:
            `location="middle"` is rejected.
        """
        with pytest.raises(ValueError, match="location must be one of"):
            add_scale_bar(ax, 100_000, ScaleBar(location="middle"))

    @pytest.mark.parametrize("bad", [-1.0, 0.0, float("nan"), float("inf")])
    def test_bad_length_raises(self, ax, bad):
        """A non-finite or non-positive `length` raises.

        Args:
            ax: The axes fixture.
            bad: An invalid length.

        Test scenario:
            Zero, negative, NaN and inf lengths are all rejected.
        """
        with pytest.raises(ValueError, match="length must be a finite positive"):
            add_scale_bar(ax, bad)

    def test_bad_pad_raises(self, ax):
        """An out-of-range `pad` raises.

        Test scenario:
            `pad=1.5` is outside `[0, 1)`.
        """
        with pytest.raises(ValueError, match="margin must be in"):
            add_scale_bar(ax, 100_000, ScaleBar(pad=1.5))

    def test_segments_below_one_raises(self, ax):
        """`segments < 1` raises.

        Test scenario:
            Zero segments is rejected.
        """
        with pytest.raises(ValueError, match="segments must be >= 1"):
            add_scale_bar(ax, 100_000, ScaleBar(segments=0))

    def test_bad_label_location_raises(self, ax):
        """An unknown `label_location` raises.

        Test scenario:
            `label_location="left"` is rejected.
        """
        with pytest.raises(ValueError, match="label_location must be"):
            add_scale_bar(ax, 100_000, ScaleBar(label_location="left"))

    def test_zero_width_range_raises(self, ax):
        """A zero-width x-range raises.

        Test scenario:
            A degenerate x-limit (matplotlib normally auto-expands identical
            limits, so it is forced here) cannot size a bar.
        """
        ax.get_xlim = lambda: (5.0, 5.0)
        with pytest.raises(ValueError, match="zero-width x-range"):
            add_scale_bar(ax, 1.0)

    def test_bar_too_wide_raises(self, ax):
        """A bar wider than the axes (with pad) raises.

        Test scenario:
            A 600000-unit bar on a 500000 range does not fit.
        """
        with pytest.raises(ValueError, match="does not fit"):
            add_scale_bar(ax, 600_000)


class TestNorthArrowPatches:
    """Tests for the `_north_arrow_patches` / `_rose_base` helpers."""

    @pytest.mark.parametrize("style, count", [("arrow", 1), ("needle", 4), ("rose", 8)])
    def test_patch_count(self, style, count):
        """Each style produces its expected number of polygons.

        Args:
            style: The arrow style.
            count: The expected polygon count.

        Test scenario:
            arrow -> 1, needle -> 4 (two halves x two ends), rose -> 8.
        """
        patches = _north_arrow_patches(style, "black", "white")
        assert len(patches) == count, f"{style} should yield {count} polygons"
        assert all(isinstance(p, Polygon) for p in patches), "all Polygons"

    def test_rose_base_sides_differ(self):
        """The two flanks of a rose ray differ.

        Test scenario:
            The left and right base vertices of the north ray are distinct.
        """
        assert _rose_base(0, 0.5, "l") != _rose_base(0, 0.5, "r"), "flanks differ"


class TestAddNorthArrow:
    """Tests for `add_north_arrow`."""

    @pytest.mark.parametrize("style", list(_NORTH_STYLES))
    def test_styles_render(self, ax, style):
        """Every style draws at least one polygon on the inset.

        Args:
            ax: The axes fixture.
            style: The style under test.

        Test scenario:
            The returned inset carries the style's polygons.
        """
        arrow = add_north_arrow(ax, spec=NorthArrow(style=style))
        assert isinstance(arrow, Axes), "returns the inset axes"
        assert len(arrow.patches) >= 1, f"{style} should draw polygons"

    @pytest.mark.parametrize(
        "location, at_right, at_top",
        [
            ("lower left", False, False),
            ("lower right", True, False),
            ("upper left", False, True),
            ("upper right", True, True),
        ],
    )
    def test_corner_placement(self, ax, location, at_right, at_top):
        """The arrow anchors in each of the four corners.

        Args:
            ax: The axes fixture.
            location: The corner under test.
            at_right: Whether the corner is on the right.
            at_top: Whether the corner is on the top.

        Test scenario:
            The inset origin sits `pad` from the expected edges.
        """
        arrow = add_north_arrow(ax, spec=NorthArrow(location=location, pad=0.03))
        x0, y0, w, h = _axfrac(ax, arrow)
        assert x0 == pytest.approx(1.0 - 0.03 - w if at_right else 0.03, abs=1e-6), (
            f"x0={x0}"
        )
        assert y0 == pytest.approx(1.0 - 0.03 - h if at_top else 0.03, abs=1e-6), (
            f"y0={y0}"
        )

    def test_label_default_n(self, ax):
        """The label defaults to `"N"`.

        Test scenario:
            The inset carries one text reading "N".
        """
        arrow = add_north_arrow(ax)
        assert [t.get_text() for t in arrow.texts] == ["N"], "default label is N"

    def test_label_none(self, ax):
        """`label=None` draws no label.

        Test scenario:
            The inset carries no text.
        """
        arrow = add_north_arrow(ax, spec=NorthArrow(label=None))
        assert len(arrow.texts) == 0, "no label when None"

    def test_rotation_renders(self, ax):
        """A non-zero rotation still renders the arrow.

        Test scenario:
            A 45-degree arrow draws its polygon(s).
        """
        arrow = add_north_arrow(ax, rotation=45.0, spec=NorthArrow(style="arrow"))
        assert len(arrow.patches) == 1, "rotated arrow still draws"

    def test_box_draws_panel(self, ax):
        """`box` adds a backing rectangle on the parent axes.

        Test scenario:
            The parent gains one `Rectangle`.
        """
        before = len(ax.patches)
        add_north_arrow(ax, spec=NorthArrow(box=True))
        assert len(ax.patches) == before + 1, "one backing panel"

    def test_zorder_above_data(self, ax):
        """The arrow sits above the data.

        Test scenario:
            The inset zorder exceeds the image zorder.
        """
        img_z = ax.get_images()[0].get_zorder()
        arrow = add_north_arrow(ax)
        assert arrow.get_zorder() > img_z, "arrow above the data"

    def test_bad_location_raises(self, ax):
        """An unknown `location` raises.

        Test scenario:
            `location="center"` is rejected.
        """
        with pytest.raises(ValueError, match="location must be one of"):
            add_north_arrow(ax, spec=NorthArrow(location="center"))

    def test_bad_style_raises(self, ax):
        """An unknown `style` raises.

        Test scenario:
            `style="compass"` is rejected.
        """
        with pytest.raises(ValueError, match="style must be one of"):
            add_north_arrow(ax, spec=NorthArrow(style="compass"))

    def test_non_finite_rotation_raises(self, ax):
        """A non-finite `rotation` raises.

        Test scenario:
            NaN rotation is rejected.
        """
        with pytest.raises(ValueError, match="rotation must be a finite"):
            add_north_arrow(ax, rotation=float("nan"))

    def test_too_big_raises(self, ax):
        """An arrow larger than the axes (with pad) raises.

        Test scenario:
            `size=1.5` does not fit.
        """
        with pytest.raises(ValueError, match="leaves no room"):
            add_north_arrow(ax, spec=NorthArrow(size=1.5))


class TestGeoMixinFurniture:
    """Tests for the `GeoMixin.add_scale_bar` / `add_north_arrow` sugar."""

    def test_scale_bar_sugar(self):
        """`glyph.add_scale_bar` draws on the glyph's axes.

        Test scenario:
            An ArrayGlyph plotted then decorated gains a segmented bar.
        """
        glyph = ArrayGlyph(
            np.arange(100.0).reshape(10, 10), extent=[0, 0, 500_000, 500_000]
        )
        glyph.plot()
        bar = glyph.add_scale_bar(100_000, ScaleBar(segments=3, label="100 km"))
        assert isinstance(bar, Axes), "sugar returns an Axes"
        assert len(bar.patches) == 3, "sugar draws the three blocks"
        plt.close(glyph.fig)

    def test_north_arrow_sugar(self):
        """`glyph.add_north_arrow` draws on the glyph's axes.

        Test scenario:
            An ArrayGlyph plotted then decorated gains a north arrow.
        """
        glyph = ArrayGlyph(
            np.arange(100.0).reshape(10, 10), extent=[0, 0, 500_000, 500_000]
        )
        glyph.plot()
        arrow = glyph.add_north_arrow(rotation=10.0, spec=NorthArrow(style="needle"))
        assert isinstance(arrow, Axes), "sugar returns an Axes"
        assert len(arrow.patches) == 4, "sugar draws the needle polygons"
        plt.close(glyph.fig)
