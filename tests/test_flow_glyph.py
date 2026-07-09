"""Tests for cleopatra.flow_glyph.FlowGlyph and styles.width_legend — issue #157.

Covers the width legend helper, FlowGlyph construction/validation, the
value→width resolution, the width-legend drawing helper, and the `plot`
rendering contract (colour by value, width by magnitude, colorbar toggle).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.legend import Legend

from cleopatra.flow_glyph import FLOW_DEFAULT_OPTIONS, FlowGlyph
from cleopatra.styles import width_legend


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


@pytest.fixture()
def paths():
    """Five horizontal polylines stacked vertically.

    Returns:
        list[np.ndarray]: Five `(2, 2)` vertex arrays.
    """
    return [np.array([[0.0, float(i)], [1.0, float(i)]]) for i in range(5)]


@pytest.fixture()
def values():
    """A per-path colour magnitude aligned with the `paths` fixture.

    Returns:
        np.ndarray: Five ascending values.
    """
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture()
def widths():
    """A per-path width magnitude (deliberately unsorted).

    Returns:
        np.ndarray: Five magnitudes whose ranking differs from `values`.
    """
    return np.array([2.0, 1.0, 5.0, 3.0, 9.0])


class TestWidthLegend:
    """Tests for the styles.width_legend helper."""

    def test_labels_round_trip(self):
        """The legend exposes its labels in order.

        Test scenario:
            Three labels come back unchanged from the legend texts.
        """
        fig, ax = plt.subplots()
        legend = width_legend(ax, [1.0, 3.0, 5.0], ["low", "mid", "high"])
        texts = [t.get_text() for t in legend.get_texts()]
        assert texts == ["low", "mid", "high"], f"Unexpected labels: {texts}"

    def test_linewidth_matches_input(self):
        """Each proxy line uses the supplied linewidth.

        Test scenario:
            Inputs 1 and 4 become handle linewidths 1 and 4.
        """
        fig, ax = plt.subplots()
        legend = width_legend(ax, [1.0, 4.0], ["thin", "thick"])
        handles = legend.legend_handles
        assert handles[0].get_linewidth() == 1.0, "First handle should be 1.0 wide"
        assert handles[1].get_linewidth() == 4.0, "Second handle should be 4.0 wide"

    def test_forwards_legend_kwargs(self):
        """`Axes.legend` kwargs such as `title` are forwarded.

        Test scenario:
            A title passed through reaches the legend title text.
        """
        fig, ax = plt.subplots()
        legend = width_legend(ax, [1.0, 2.0], ["a", "b"], title="Flow")
        assert (
            legend.get_title().get_text() == "Flow"
        ), "title kwarg should be forwarded"

    def test_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError.

        Test scenario:
            Two widths but one label is rejected.
        """
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="same length"):
            width_legend(ax, [1.0, 2.0], ["only-one"])


class TestFlowGlyphInit:
    """Tests for FlowGlyph.__init__."""

    def test_paths_stored_as_float_arrays(self, paths):
        """Path vertices are stored as float ndarrays.

        Test scenario:
            Integer-typed vertex lists are converted to float arrays.
        """
        glyph = FlowGlyph([[[0, 0], [1, 1]]])
        assert glyph.paths[0].dtype == float, "Paths should be stored as float arrays"
        assert (
            glyph.values is None and glyph.widths is None
        ), "values/widths default to None"

    def test_defaults_present(self, paths):
        """Flow-specific option defaults are exposed.

        Test scenario:
            width_limits/width_scale defaults and None ticks_spacing.
        """
        glyph = FlowGlyph(paths)
        assert glyph.default_options["width_limits"] == (
            1,
            5,
        ), "Default width_limits should be (1, 5)"
        assert (
            glyph.default_options["width_scale"] == "linear"
        ), "Default width_scale should be linear"
        assert (
            glyph.default_options["ticks_spacing"] is None
        ), "ticks_spacing should auto-derive"

    def test_mismatched_values_raises(self, paths):
        """A values array not matching the path count raises ValueError.

        Test scenario:
            len(values) != len(paths) is rejected.
        """
        with pytest.raises(ValueError, match="values must have one entry per path"):
            FlowGlyph(paths, values=np.array([1.0, 2.0]))

    def test_mismatched_widths_raises(self, paths):
        """A widths array not matching the path count raises ValueError.

        Test scenario:
            len(widths) != len(paths) is rejected.
        """
        with pytest.raises(ValueError, match="widths must have one entry per path"):
            FlowGlyph(paths, widths=np.array([1.0, 2.0]))

    def test_invalid_kwarg_raises(self, paths):
        """An unknown kwarg is rejected by the strict merge.

        Test scenario:
            A key absent from FLOW_DEFAULT_OPTIONS raises ValueError.
        """
        with pytest.raises(ValueError, match="not correct"):
            FlowGlyph(paths, not_an_option=1)


class TestFlowGlyphResolveLinewidths:
    """Tests for FlowGlyph._resolve_linewidths."""

    def test_scalar_when_no_widths(self, paths):
        """Without widths the scalar `line_width` option is returned.

        Test scenario:
            A FlowGlyph with no widths resolves to the scalar line_width.
        """
        glyph = FlowGlyph(paths, line_width=2.5)
        assert glyph._resolve_linewidths() == 2.5, "Should return scalar line_width"

    def test_array_spans_width_limits(self, paths, widths):
        """With widths the result spans width_limits monotonically.

        Test scenario:
            The per-path widths run from width_limits[0] to width_limits[1]
            in the order of the input magnitudes.
        """
        glyph = FlowGlyph(paths, widths=widths, width_limits=(1, 5))
        out = np.asarray(glyph._resolve_linewidths())
        assert np.isclose(out.min(), 1.0), f"min width should be 1.0, got {out.min()}"
        assert np.isclose(out.max(), 5.0), f"max width should be 5.0, got {out.max()}"
        assert np.array_equal(
            np.argsort(out), np.argsort(widths)
        ), "Widths not monotonic in input"

    @pytest.mark.parametrize("scale", ["linear", "log", "sqrt"])
    def test_width_scale_modes(self, paths, scale):
        """Each width_scale yields a monotonic per-path width array.

        Args:
            paths: The polyline fixture.
            scale: The width scale under test.

        Test scenario:
            For positive magnitudes every scale preserves the ranking.
        """
        widths = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        glyph = FlowGlyph(paths, widths=widths, width_scale=scale, width_limits=(1, 5))
        out = np.asarray(glyph._resolve_linewidths())
        assert np.array_equal(
            np.argsort(out), np.argsort(widths)
        ), f"{scale} not monotonic"


class TestFlowGlyphDrawWidthLegend:
    """Tests for FlowGlyph._draw_width_legend (via plot)."""

    def test_default_three_quantile_entries(self, paths, widths):
        """The default width legend has three (min/median/max) entries.

        Test scenario:
            size_legend=True with no explicit values draws three entries.
        """
        glyph = FlowGlyph(paths, widths=widths, size_legend=True)
        glyph.plot()
        assert isinstance(
            glyph.size_legend_artist, Legend
        ), "A width legend should be drawn"
        texts = [t.get_text() for t in glyph.size_legend_artist.get_texts()]
        assert len(texts) == 3, f"Default legend should have 3 entries, got {texts}"

    def test_explicit_legend_values(self, paths, widths):
        """Explicit size_legend_values control the legend entries.

        Test scenario:
            Two representative magnitudes give two labelled entries.
        """
        glyph = FlowGlyph(
            paths, widths=widths, size_legend=True, size_legend_values=[2.0, 8.0]
        )
        glyph.plot()
        texts = [t.get_text() for t in glyph.size_legend_artist.get_texts()]
        assert texts == ["2", "8"], f"Legend should honour explicit values, got {texts}"

    def test_no_legend_without_widths(self, paths, values):
        """No legend is drawn when there are no widths, even if requested.

        Test scenario:
            size_legend=True but widths None leaves the legend unset.
        """
        glyph = FlowGlyph(paths, values=values, size_legend=True)
        glyph.plot()
        assert glyph.size_legend_artist is None, "No width legend without widths"


class TestFlowGlyphPlot:
    """Tests for FlowGlyph.plot."""

    def test_returns_line_collection(self, paths):
        """plot returns a LineCollection added to the axes.

        Test scenario:
            The third return value is a LineCollection.
        """
        glyph = FlowGlyph(paths)
        fig, ax, lc = glyph.plot()
        assert isinstance(
            lc, LineCollection
        ), f"Expected LineCollection, got {type(lc)}"

    def test_uncoloured_single_colour_no_colorbar(self, paths):
        """Without values the lines are single-colour with no colorbar.

        Test scenario:
            values None -> no colour array and no colorbar.
        """
        glyph = FlowGlyph(paths)
        _, _, lc = glyph.plot()
        assert lc.get_array() is None, "Uncoloured collection should carry no array"
        assert glyph.cbar is None, "No colorbar without values"

    def test_coloured_carries_values_and_colorbar(self, paths, values):
        """With values the collection carries them and gains a colorbar.

        Test scenario:
            get_array equals the input values and a colorbar is attached.
        """
        glyph = FlowGlyph(paths, values=values)
        _, _, lc = glyph.plot()
        assert np.array_equal(
            lc.get_array(), values
        ), "Colour array should equal values"
        assert glyph.cbar is not None, "A colorbar should be drawn by default"

    def test_colorbar_reflects_value_range(self, paths, values):
        """The colorbar spans the value range.

        Test scenario:
            The colorbar starts at the value minimum and covers the maximum
            (its upper limit may round slightly past `vmax` via the shared
            tick spacing, exactly as for the other glyphs).
        """
        glyph = FlowGlyph(paths, values=values)
        glyph.plot()
        lo, hi = glyph.cbar.mappable.get_clim()
        assert np.isclose(
            lo, values.min()
        ), f"Colorbar should start at {values.min()}, got {lo}"
        assert hi >= values.max(), f"Colorbar should cover {values.max()}, got {hi}"

    def test_colorbar_with_discrete_levels(self, paths, values):
        """An integer `levels` colours via a discrete BoundaryNorm.

        Test scenario:
            With levels set, the collection's norm carries discrete
            boundaries (and `set_clim` is bypassed).
        """
        glyph = FlowGlyph(paths, values=values, levels=4)
        _, _, lc = glyph.plot()
        assert lc.norm.boundaries is not None, "levels should yield a BoundaryNorm"
        assert glyph.cbar is not None, "A colorbar should still be drawn"

    def test_linewidths_monotonic_and_span_limits(self, paths, widths):
        """get_linewidths runs monotonically with widths and spans limits.

        Test scenario:
            The drawn linewidths order matches the widths order and the
            extremes equal width_limits.
        """
        glyph = FlowGlyph(paths, widths=widths, width_limits=(1, 5))
        _, _, lc = glyph.plot()
        lw = np.asarray(lc.get_linewidths())
        assert np.array_equal(
            np.argsort(lw), np.argsort(widths)
        ), "Linewidths not monotonic in widths"
        assert np.isclose(lw.min(), 1.0) and np.isclose(
            lw.max(), 5.0
        ), f"Linewidths should span (1, 5), got [{lw.min()}, {lw.max()}]"

    def test_add_colorbar_false_suppresses(self, paths, values):
        """add_colorbar=False suppresses the colorbar.

        Test scenario:
            The plot-time override wins and no colorbar is created.
        """
        glyph = FlowGlyph(paths, values=values)
        glyph.plot(add_colorbar=False)
        assert glyph.cbar is None, "add_colorbar=False should suppress the colorbar"

    def test_widths_none_uses_scalar_line_width(self, paths, values):
        """Without widths every line uses the scalar line_width.

        Test scenario:
            All linewidths equal the line_width option.
        """
        glyph = FlowGlyph(paths, values=values, line_width=2.0)
        _, _, lc = glyph.plot()
        lw = np.asarray(lc.get_linewidths())
        assert np.all(
            lw == 2.0
        ), f"All linewidths should be 2.0, got {set(lw.tolist())}"

    def test_title_override(self, paths):
        """A title passed to plot overrides the default.

        Test scenario:
            plot(title=...) sets the axes title.
        """
        glyph = FlowGlyph(paths)
        _, ax, _ = glyph.plot(title="Flows")
        assert (
            ax.get_title() == "Flows"
        ), f"Title should be 'Flows', got {ax.get_title()!r}"

    def test_plot_on_supplied_axes(self, paths):
        """plot(ax=...) draws on the supplied axes.

        Test scenario:
            A pre-existing axes is used and its figure adopted.
        """
        fig, host_ax = plt.subplots()
        glyph = FlowGlyph(paths)
        _, out_ax, _ = glyph.plot(ax=host_ax)
        assert out_ax is host_ax, "plot should draw on the supplied axes"

    def test_plot_uses_axes_from_construction(self, paths):
        """plot() reuses an axes supplied at construction.

        Test scenario:
            With an axes passed to __init__, a no-arg plot() reuses it.
        """
        fig, host_ax = plt.subplots()
        glyph = FlowGlyph(paths, ax=host_ax)
        _, out_ax, _ = glyph.plot()
        assert out_ax is host_ax, "plot() should reuse the construction axes"
