"""Tests for cleopatra.glyphs.primitives.polygon_glyph.PolygonGlyph (T7.2c)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection

from cleopatra.glyphs.primitives.polygon_glyph import (
    OUTLINE_EDGECOLOR,
    POLYGON_DEFAULT_OPTIONS,
    PolygonGlyph,
)
from cleopatra.styling.params import Contour


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


@pytest.fixture()
def polygons():
    """Two adjacent triangles as (n, 2) vertex arrays.

    Returns:
        list[np.ndarray]: Two triangular polygons.
    """
    return [
        np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
        np.array([[1.0, 0.0], [2.0, 0.0], [1.5, 1.0]]),
    ]


class TestPolygonGlyphInit:
    """Tests for PolygonGlyph.__init__."""

    def test_stores_polygons_as_float_arrays(self, polygons):
        """Polygons are coerced to float ndarrays.

        Test scenario:
            Integer-vertex input is stored as float arrays.
        """
        glyph = PolygonGlyph([[[0, 0], [1, 0], [0, 1]]])
        assert glyph.polygons[0].dtype == float, "Vertices should be float"
        assert glyph.values is None, "values should default to None"

    def test_options_include_polygon_keys(self, polygons):
        """Polygon-specific option keys are present with defaults.

        Test scenario:
            edgecolor/linewidth defaults exist and ticks_spacing is None.
        """
        glyph = PolygonGlyph(polygons)
        assert glyph.default_options["edgecolor"] == "none", "Default edgecolor 'none'"
        assert glyph.default_options["linewidth"] == 0.5, "Default linewidth 0.5"
        assert glyph.default_options["ticks_spacing"] is None, (
            "ticks_spacing should default to None for auto-derivation"
        )

    def test_mismatched_values_raises(self, polygons):
        """A values array not matching polygon count raises ValueError.

        Test scenario:
            len(values) != number of polygons is rejected.
        """
        with pytest.raises(ValueError, match="must match the number"):
            PolygonGlyph(polygons, values=np.array([1.0, 2.0, 3.0]))

    def test_facecolor_and_values_mutually_exclusive(self, polygons):
        """Passing both facecolor and values raises ValueError.

        Test scenario:
            Without this guard, facecolor was silently dropped in favour of
            the values-driven colormap path -- the same kind of conflict
            ScatterGlyph's colors/values and flat_color/values guard
            against, so facecolor/values must raise too rather than resolve
            one option in silence.
        """
        with pytest.raises(ValueError, match="mutually exclusive"):
            PolygonGlyph(polygons, values=np.array([10.0, 20.0]), facecolor="red")

    def test_2d_values_with_matching_row_count_raises(self, polygons):
        """A 2-D values array is rejected even when its row count matches.

        Test scenario:
            An accidentally un-flattened `(N, 2)` values array has the right
            `shape[0]` but the wrong `ndim`; the full-shape check must still
            reject it rather than silently mis-sizing the colour array once
            it is later flattened.
        """
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert values.shape[0] == len(polygons), "Fixture assumption: row count matches"
        with pytest.raises(ValueError, match="must match the number"):
            PolygonGlyph(polygons, values=values)

    def test_invalid_kwarg_raises(self, polygons):
        """An unknown kwarg is rejected by the strict merge.

        Test scenario:
            A key absent from POLYGON_DEFAULT_OPTIONS raises ValueError.
        """
        with pytest.raises(ValueError, match="not correct"):
            PolygonGlyph(polygons, bogus=1)


class TestPolygonGlyphPlot:
    """Tests for PolygonGlyph.plot."""

    def test_filled_maps_values_and_adds_colorbar(self, polygons):
        """Filled polygons carry the value array and gain a colorbar.

        Test scenario:
            The PolyCollection array equals the values and a colorbar
            is attached.
        """
        glyph = PolygonGlyph(polygons, values=np.array([10.0, 20.0]))
        fig, ax, pc = glyph.plot()
        assert isinstance(pc, PolyCollection), (
            f"Expected PolyCollection, got {type(pc)}"
        )
        np.testing.assert_array_almost_equal(
            pc.get_array(), [10.0, 20.0], err_msg="Array should equal values"
        )
        assert glyph.cbar is not None, (
            "A colorbar should be attached for filled polygons"
        )
        assert pc in ax.collections, "The collection should be added to the axes"

    def test_no_values_is_outline(self, polygons):
        """With no values, only outlines are drawn (no colour, no cbar).

        Test scenario:
            No values -> PolyCollection without an array and no colorbar.
        """
        glyph = PolygonGlyph(polygons)
        _, _, pc = glyph.plot()
        assert pc.get_array() is None, "Outline collection should carry no array"
        assert glyph.cbar is None, "No colorbar without values"

    def test_outline_only_overrides_values(self, polygons):
        """outline_only draws outlines even when values are present.

        Test scenario:
            outline_only=True suppresses fill/colorbar despite values.
        """
        glyph = PolygonGlyph(polygons, values=np.array([10.0, 20.0]))
        _, _, pc = glyph.plot(outline_only=True)
        assert pc.get_array() is None, "outline_only should suppress the colour array"
        assert glyph.cbar is None, "outline_only should suppress the colorbar"

    def test_outline_only_edges_are_visible_by_default(self, polygons):
        """Outline mode substitutes a visible edge for the 'none' default.

        Test scenario:
            The default edgecolor is 'none' (borderless fill), which would
            render an unfilled polygon invisible; outline mode must fall
            back to OUTLINE_EDGECOLOR so the outlines are actually drawn.
        """
        glyph = PolygonGlyph(polygons)
        _, _, pc = glyph.plot(outline_only=True)
        edgecolors = pc.get_edgecolor()
        assert len(edgecolors) > 0, "outline mode should set at least one edge colour"
        assert np.allclose(edgecolors[0], mcolors.to_rgba(OUTLINE_EDGECOLOR)), (
            f"expected the {OUTLINE_EDGECOLOR!r} fallback, got {edgecolors[0]}"
        )
        assert np.allclose(pc.get_facecolor(), np.empty((0, 4))), (
            "outline mode should leave the polygons unfilled"
        )

    def test_no_values_edges_are_visible_by_default(self, polygons):
        """The implicit outline path (no values) is visible too.

        Test scenario:
            Constructing without values takes the same outline branch, so
            it must also get the visible-edge fallback.
        """
        glyph = PolygonGlyph(polygons)
        _, _, pc = glyph.plot()
        assert np.allclose(pc.get_edgecolor()[0], mcolors.to_rgba(OUTLINE_EDGECOLOR)), (
            "no-values outlines should use the visible fallback edge colour"
        )

    def test_outline_only_honours_explicit_edgecolor(self, polygons):
        """An explicit edgecolor is not overridden by the fallback.

        Test scenario:
            edgecolor='navy' survives outline mode unchanged.
        """
        glyph = PolygonGlyph(polygons, edgecolor="navy", linewidth=1.5)
        _, _, pc = glyph.plot(outline_only=True)
        assert np.allclose(pc.get_edgecolor()[0], mcolors.to_rgba("navy")), (
            "an explicit edgecolor should be honoured in outline mode"
        )
        assert pc.get_linewidth()[0] == 1.5, "linewidth should be forwarded"

    def test_filled_polygons_keep_borderless_default(self, polygons):
        """The fallback does not leak into the filled branch.

        Test scenario:
            A value-filled choropleth keeps its transparent 'none' edge,
            so polygons render borderless as before.
        """
        glyph = PolygonGlyph(polygons, values=np.array([10.0, 20.0]))
        _, _, pc = glyph.plot()
        edgecolors = pc.get_edgecolor()
        assert len(edgecolors) == 0 or edgecolors[0][3] == 0.0, (
            f"filled polygons should stay borderless, got edges {edgecolors}"
        )

    def test_auto_limits_from_values(self, polygons):
        """Colour limits auto-resolve from the value range (pinned spacing).

        Test scenario:
            With a spacing that divides the range, the clim equals the
            value range.
        """
        glyph = PolygonGlyph(polygons, values=np.array([0.0, 40.0]), ticks_spacing=10.0)
        _, _, pc = glyph.plot()
        assert pc.get_clim() == (
            0.0,
            40.0,
        ), f"clim should follow the resolved limits, got {pc.get_clim()}"

    def test_levels_produce_boundary_norm(self, polygons):
        """An integer `levels` discretises the fill colour scale.

        Test scenario:
            levels=4 -> the collection norm is a BoundaryNorm.
        """
        glyph = PolygonGlyph(
            polygons, values=np.array([0.0, 10.0]), vmin=0.0, vmax=10.0
        )
        _, _, pc = glyph.plot(contour=Contour(levels=4))
        assert isinstance(pc.norm, mcolors.BoundaryNorm), (
            f"levels should yield a BoundaryNorm, got {type(pc.norm)}"
        )

    def test_edgecolor_and_linewidth_forwarded(self, polygons):
        """edgecolor/linewidth options reach the PolyCollection.

        Test scenario:
            edgecolor='black', linewidth=2 are applied to the collection.
        """
        glyph = PolygonGlyph(polygons, edgecolor="black", linewidth=2.0)
        _, _, pc = glyph.plot()
        assert np.allclose(pc.get_edgecolor()[0], mcolors.to_rgba("black")), (
            "edgecolor should be forwarded"
        )
        assert pc.get_linewidth()[0] == 2.0, "linewidth should be forwarded"

    def test_plot_on_supplied_axes(self, polygons):
        """Plotting onto a supplied axes reuses that axes/figure.

        Test scenario:
            Passing ax to plot draws on it and returns its figure.
        """
        fig, ax = plt.subplots()
        glyph = PolygonGlyph(polygons, values=np.array([1.0, 2.0]))
        out_fig, out_ax, _ = glyph.plot(ax=ax)
        assert out_ax is ax, "Should draw on the supplied axes"
        assert out_fig is fig, "Should return the supplied axes' figure"

    def test_title_override(self, polygons):
        """A title passed to plot is applied to the axes.

        Test scenario:
            plot(title=...) sets the axes title.
        """
        glyph = PolygonGlyph(polygons)
        _, ax, _ = glyph.plot(title="Regions")
        assert ax.get_title() == "Regions", f"Unexpected title: {ax.get_title()}"

    def test_all_nan_values_raise(self, polygons):
        """All-NaN values with unpinned limits raise ValueError.

        Test scenario:
            The shared helper rejects an unusable colour range.
        """
        glyph = PolygonGlyph(polygons, values=np.array([np.nan, np.nan]))
        with pytest.raises(ValueError, match="no finite values"):
            glyph.plot()

    def test_facecolor_fills_flat_colour_no_values(self, polygons):
        """facecolor fills the polygons with no values and no colorbar.

        Test scenario:
            With no values, an explicit facecolor switches from the default
            outline-only path to a flat-filled one -- no colormap involved.
        """
        glyph = PolygonGlyph(polygons, facecolor="forestgreen", alpha=0.6)
        _, _, pc = glyph.plot()
        assert pc.get_array() is None, "flat fill carries no value array"
        assert glyph.cbar is None, "flat fill draws no colorbar"
        facecolors = pc.get_facecolor()
        assert np.allclose(facecolors[:, :3], mcolors.to_rgb("forestgreen")), facecolors
        assert np.allclose(facecolors[:, 3], 0.6), facecolors

    def test_facecolor_sequence_rejected(self, polygons):
        """A facecolor that is itself a sequence of colours raises TypeError.

        Test scenario:
            `facecolor` is documented as a single flat colour. Without this
            guard, a mismatched-length list was passed straight through to
            `PolyCollection(facecolors=...)` with no error, silently
            producing a facecolor array whose length did not match the
            polygon count.
        """
        with pytest.raises(TypeError, match="single colour"):
            PolygonGlyph(polygons, facecolor=["red", "green", "blue"])

    def test_outline_only_overrides_facecolor(self, polygons):
        """outline_only wins even when a flat facecolor is set.

        Test scenario:
            An explicit outline_only=True request stays unfilled regardless
            of a construction-time facecolor.
        """
        glyph = PolygonGlyph(polygons, facecolor="forestgreen")
        _, _, pc = glyph.plot(outline_only=True)
        assert pc.get_array() is None
        assert np.allclose(pc.get_facecolor(), np.empty((0, 4))), (
            "outline_only should leave the polygons unfilled despite facecolor"
        )

    def test_alpha_forwarded_to_value_mapped_fill(self, polygons):
        """alpha applies to the scalar-mapped (colour-by-value) fill too.

        Test scenario:
            alpha is a shared opacity option, not exclusive to the flat-fill
            path -- it also reaches the value-mapped PolyCollection.
        """
        glyph = PolygonGlyph(polygons, values=np.array([10.0, 20.0]), alpha=0.5)
        _, _, pc = glyph.plot()
        assert pc.get_array() is not None
        assert np.allclose(pc.get_alpha(), 0.5), pc.get_alpha()

    def test_grouped_color_rejects_plain_string(self, polygons):
        """plot(color=<str>) still raises -- color= stays a grouped param.

        Test scenario:
            facecolor is a separate construction-time option, so the
            `color=` keyword to `plot` still only accepts a ColorScaling (or
            equivalent grouped object), never a loose colour string.
        """
        glyph = PolygonGlyph(polygons)
        with pytest.raises(TypeError, match="expected a grouped parameter object"):
            glyph.plot(color="navy")


def test_polygon_default_options_extend_style_defaults():
    """POLYGON_DEFAULT_OPTIONS is a superset of the shared style defaults.

    Test scenario:
        The module-level options dict carries both the base style keys
        and the polygon-specific additions.
    """
    assert "figsize" in POLYGON_DEFAULT_OPTIONS, "Should inherit base style keys"
    assert "edgecolor" in POLYGON_DEFAULT_OPTIONS, "Should add polygon keys"


class TestAddColorbarToggle:
    """`add_colorbar=False` suppresses PolygonGlyph's colorbar (#3)."""

    @staticmethod
    def _polys():
        return [np.array([[0, 0], [1, 0], [1, 1]]), np.array([[1, 1], [2, 1], [2, 2]])]

    def test_default_draws_colorbar(self):
        """A value-filled polygon layer draws its colorbar by default."""
        glyph = PolygonGlyph(self._polys(), values=np.array([1.0, 2.0]))
        fig, ax, _ = glyph.plot()
        try:
            assert glyph.cbar is not None, "default should draw a colorbar"
            assert len(fig.axes) == 2, f"expected 2 axes, got {len(fig.axes)}"
        finally:
            plt.close(fig)

    def test_add_colorbar_false_suppresses(self):
        """`add_colorbar=False` leaves cbar None and adds no axes."""
        glyph = PolygonGlyph(
            self._polys(), values=np.array([1.0, 2.0]), add_colorbar=False
        )
        fig, ax, _ = glyph.plot()
        try:
            assert glyph.cbar is None, "add_colorbar=False should skip the colorbar"
            assert len(fig.axes) == 1, f"expected 1 axes, got {len(fig.axes)}"
        finally:
            plt.close(fig)

    def test_plot_time_override_suppresses(self):
        """Passing `add_colorbar=False` to `plot` suppresses the colorbar.

        Test scenario:
            Plot-time override: even with the default construction option,
            `plot(add_colorbar=False)` draws no colorbar.
        """
        glyph = PolygonGlyph(self._polys(), values=np.array([1.0, 2.0]))
        fig, ax, _ = glyph.plot(add_colorbar=False)
        try:
            assert glyph.cbar is None, (
                "plot(add_colorbar=False) should skip the colorbar"
            )
            assert len(fig.axes) == 1, f"expected 1 axes, got {len(fig.axes)}"
        finally:
            plt.close(fig)

    def test_add_colorbar_in_option_keys(self):
        """`add_colorbar` is an accepted option key."""
        assert "add_colorbar" in PolygonGlyph.option_keys(), "add_colorbar missing"
