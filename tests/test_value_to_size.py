"""Tests for value-to-size scaling and the size legend — issue #155.

Covers:

* `cleopatra.styles.resolve_sizes` (each `size_scale`, monotonicity, range
  spanning, degenerate input, and error paths) and the `SIZE_SCALES` constant.
* `cleopatra.styles.size_legend` (area→markersize mapping, label round-trip,
  and the length-mismatch guard).
* `ScatterGlyph`'s `sizes` parameter and `size_*` options, including the
  `_resolve_marker_area` / `_draw_size_legend` helpers and the `sizes=None`
  regression.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.legend import Legend

from cleopatra.scatter_glyph import ScatterGlyph
from cleopatra.styles import SIZE_SCALES, resolve_sizes, size_legend


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


@pytest.fixture()
def xy():
    """Five points on a horizontal line.

    Returns:
        tuple[np.ndarray, np.ndarray]: x (0..4) and y (zeros) arrays.
    """
    x = np.arange(5.0)
    return x, np.zeros_like(x)


class TestResolveSizes:
    """Tests for the ``resolve_sizes`` value→size helper."""

    def test_size_scales_constant(self):
        """``SIZE_SCALES`` enumerates exactly the supported transforms.

        Test scenario:
            The accepted scales are linear, log, and sqrt.
        """
        assert set(SIZE_SCALES) == {"linear", "log", "sqrt"}, (
            f"Unexpected SIZE_SCALES: {SIZE_SCALES}"
        )

    def test_linear_spans_output_range(self):
        """Linear scaling maps min→out_min and max→out_max.

        Test scenario:
            A 0..10 ramp maps onto [10, 200] with the midpoint centred.
        """
        sizes = resolve_sizes(np.array([0.0, 5.0, 10.0]), 10.0, 200.0)
        assert np.allclose(sizes, [10.0, 105.0, 200.0]), f"Unexpected sizes: {sizes}"

    @pytest.mark.parametrize("scale", ["linear", "log", "sqrt"])
    def test_monotonic_in_input(self, scale):
        """Every scale preserves the ranking of the input magnitudes.

        Args:
            scale: The size scale under test.

        Test scenario:
            The argsort of the output equals the argsort of the input, so
            larger magnitudes always map to larger sizes.
        """
        values = np.array([3.0, 1.0, 2.0, 8.0, 5.0])
        sizes = resolve_sizes(values, 10.0, 200.0, scale=scale)
        assert np.array_equal(np.argsort(sizes), np.argsort(values)), (
            f"{scale} mapping is not monotonic: {sizes}"
        )

    @pytest.mark.parametrize("scale", ["linear", "log", "sqrt"])
    def test_spans_exactly_the_limits(self, scale):
        """The smallest/largest outputs equal out_min/out_max for each scale.

        Args:
            scale: The size scale under test.

        Test scenario:
            With positive, non-degenerate input the output min and max are
            exactly the requested limits.
        """
        values = np.array([1.0, 2.0, 4.0, 8.0])
        sizes = resolve_sizes(values, 5.0, 50.0, scale=scale)
        assert np.isclose(sizes.min(), 5.0), f"min should be 5.0, got {sizes.min()}"
        assert np.isclose(sizes.max(), 50.0), f"max should be 50.0, got {sizes.max()}"

    def test_log_compresses_orders_of_magnitude(self):
        """Log scaling makes evenly-spaced decades evenly-spaced sizes.

        Test scenario:
            [1, 10, 100] over [0, 1] maps to [0, 0.5, 1] (log10 is linear in
            decades).
        """
        sizes = resolve_sizes(np.array([1.0, 10.0, 100.0]), 0.0, 1.0, scale="log")
        assert np.allclose(sizes, [0.0, 0.5, 1.0]), f"Unexpected log sizes: {sizes}"

    def test_sqrt_uses_area_perception(self):
        """Sqrt scaling places 4 at the midpoint of [0, 4]→sizes.

        Test scenario:
            sqrt([0, 1, 4]) = [0, 1, 2] maps onto [0, 1] as [0, 0.5, 1].
        """
        sizes = resolve_sizes(np.array([0.0, 1.0, 4.0]), 0.0, 1.0, scale="sqrt")
        assert np.allclose(sizes, [0.0, 0.5, 1.0]), f"Unexpected sqrt sizes: {sizes}"

    def test_all_equal_maps_to_midpoint(self):
        """Constant input maps every item to the output midpoint.

        Test scenario:
            No spread to encode -> each size is (out_min + out_max) / 2.
        """
        sizes = resolve_sizes(np.full(4, 7.0), 10.0, 50.0)
        assert np.allclose(sizes, 30.0), f"Constant input should map to 30.0, got {sizes}"

    def test_preserves_input_shape(self):
        """The output has the same shape as the input array.

        Test scenario:
            A length-5 input yields a length-5 output.
        """
        sizes = resolve_sizes(np.arange(5.0), 1.0, 2.0)
        assert sizes.shape == (5,), f"Expected shape (5,), got {sizes.shape}"

    def test_unknown_scale_raises(self):
        """An unrecognised ``scale`` raises ``ValueError`` listing valid ones.

        Test scenario:
            "cubic" is not a supported size scale.
        """
        with pytest.raises(ValueError, match="Invalid size_scale"):
            resolve_sizes(np.arange(5.0), 1.0, 2.0, scale="cubic")

    def test_log_non_positive_raises(self):
        """``scale='log'`` with a non-positive magnitude raises ``ValueError``.

        Test scenario:
            A zero magnitude cannot be log-scaled.
        """
        with pytest.raises(ValueError, match="strictly positive"):
            resolve_sizes(np.array([0.0, 1.0, 2.0]), 1.0, 2.0, scale="log")

    def test_sqrt_negative_raises(self):
        """``scale='sqrt'`` with a negative magnitude raises ``ValueError``.

        Test scenario:
            A negative magnitude cannot be sqrt-scaled.
        """
        with pytest.raises(ValueError, match="non-negative"):
            resolve_sizes(np.array([-1.0, 1.0, 2.0]), 1.0, 2.0, scale="sqrt")

    def test_all_non_finite_raises(self):
        """All-non-finite input raises a clear ``ValueError``.

        Test scenario:
            No finite magnitude is available to define the domain.
        """
        with pytest.raises(ValueError, match="no finite entries"):
            resolve_sizes(np.array([np.nan, np.inf]), 1.0, 2.0)


class TestSizeLegend:
    """Tests for the ``size_legend`` helper."""

    def test_labels_round_trip(self):
        """The legend exposes the labels it was given, in order.

        Test scenario:
            Three labels come back unchanged from the legend texts.
        """
        fig, ax = plt.subplots()
        legend = size_legend(ax, [20.0, 100.0, 200.0], ["low", "mid", "high"])
        texts = [t.get_text() for t in legend.get_texts()]
        assert texts == ["low", "mid", "high"], f"Unexpected legend labels: {texts}"

    def test_marker_size_is_sqrt_of_area(self):
        """Proxy ``markersize`` is the square root of the supplied area.

        Test scenario:
            Areas 16 and 64 (points²) become markersizes 4 and 8 (points).
        """
        fig, ax = plt.subplots()
        legend = size_legend(ax, [16.0, 64.0], ["small", "big"])
        handles = legend.legend_handles
        assert np.isclose(handles[0].get_markersize(), 4.0), "16 -> markersize 4"
        assert np.isclose(handles[1].get_markersize(), 8.0), "64 -> markersize 8"

    def test_returns_legend_attached_to_axes(self):
        """The returned legend is the one registered on the axes.

        Test scenario:
            ``ax.get_legend()`` is the object returned by ``size_legend``.
        """
        fig, ax = plt.subplots()
        legend = size_legend(ax, [10.0, 20.0], ["a", "b"])
        assert ax.get_legend() is legend, "Legend should be attached to the axes"

    def test_forwards_legend_kwargs(self):
        """``Axes.legend`` kwargs such as ``title`` are forwarded.

        Test scenario:
            A ``title`` passed through reaches the legend title text.
        """
        fig, ax = plt.subplots()
        legend = size_legend(ax, [10.0, 20.0], ["a", "b"], title="Population")
        assert legend.get_title().get_text() == "Population", "title kwarg should be forwarded"

    def test_negative_area_clamped(self):
        """A negative area is clamped to zero markersize (no math error).

        Test scenario:
            ``sqrt`` of a clamped negative area is 0, not NaN.
        """
        fig, ax = plt.subplots()
        legend = size_legend(ax, [-4.0, 16.0], ["neg", "pos"])
        handles = legend.legend_handles
        assert handles[0].get_markersize() == 0.0, "Negative area should clamp to 0"

    def test_length_mismatch_raises(self):
        """Mismatched ``marker_sizes`` / ``labels`` lengths raise ``ValueError``.

        Test scenario:
            Two sizes but one label is rejected.
        """
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="same length"):
            size_legend(ax, [10.0, 20.0], ["only-one"])


class TestScatterGlyphSizes:
    """Integration tests for the ``sizes`` parameter and ``size_*`` options."""

    def test_sizes_span_limits_monotonically(self, xy):
        """A ``sizes`` array yields per-point areas spanning ``size_limits``.

        Test scenario:
            ``get_sizes`` runs monotonically with the input and its extremes
            equal the configured ``size_limits``.
        """
        x, y = xy
        glyph = ScatterGlyph(x, y, sizes=np.array([2.0, 1.0, 5.0, 3.0, 9.0]),
                             size_limits=(10, 200))
        _, _, paths = glyph.plot()
        out = paths.get_sizes()
        assert np.isclose(out.min(), 10.0), f"min area should be 10, got {out.min()}"
        assert np.isclose(out.max(), 200.0), f"max area should be 200, got {out.max()}"
        assert np.array_equal(
            np.argsort(out), np.argsort([2.0, 1.0, 5.0, 3.0, 9.0])
        ), f"Areas not monotonic in input: {out}"

    def test_size_scale_option_applied(self, xy):
        """The ``size_scale`` option drives the transform used.

        Test scenario:
            With ``size_scale='log'`` and decade-spaced input, the middle
            point sits at the midpoint of the limits.
        """
        x, y = xy[0][:3], xy[1][:3]
        glyph = ScatterGlyph(x, y, sizes=np.array([1.0, 10.0, 100.0]),
                             size_limits=(0, 100), size_scale="log")
        _, _, paths = glyph.plot()
        out = paths.get_sizes()
        assert np.isclose(out[1], 50.0), f"log midpoint should be 50, got {out[1]}"

    def test_size_legend_renders_default_entries(self, xy):
        """``size_legend=True`` draws a three-entry legend by default.

        Test scenario:
            With no explicit values, the min/median/max quantiles give three
            legend entries stored on ``size_legend_artist``.
        """
        x, y = xy
        glyph = ScatterGlyph(x, y, sizes=np.arange(1.0, 6.0), size_legend=True)
        glyph.plot()
        assert isinstance(glyph.size_legend_artist, Legend), "A size legend should be drawn"
        texts = [t.get_text() for t in glyph.size_legend_artist.get_texts()]
        assert len(texts) == 3, f"Default legend should have 3 entries, got {texts}"

    def test_size_legend_values_honoured(self, xy):
        """Explicit ``size_legend_values`` control the legend entries.

        Test scenario:
            Two representative magnitudes give two labelled legend entries.
        """
        x, y = xy
        glyph = ScatterGlyph(x, y, sizes=np.arange(1.0, 6.0), size_legend=True,
                             size_legend_values=[2.0, 4.0])
        glyph.plot()
        texts = [t.get_text() for t in glyph.size_legend_artist.get_texts()]
        assert texts == ["2", "4"], f"Legend should honour explicit values, got {texts}"

    def test_no_legend_when_disabled(self, xy):
        """No size legend is drawn when ``size_legend`` is False.

        Test scenario:
            Sizes given but ``size_legend`` left at its default False.
        """
        x, y = xy
        glyph = ScatterGlyph(x, y, sizes=np.arange(1.0, 6.0))
        glyph.plot()
        assert glyph.size_legend_artist is None, "No legend should be drawn when disabled"

    def test_color_and_size_independent(self, xy):
        """Colour (``values``) and size (``sizes``) combine independently.

        Test scenario:
            The scatter carries the raw colour array AND size-scaled areas at
            once.
        """
        x, y = xy
        values = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        sizes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        glyph = ScatterGlyph(x, y, values=values, sizes=sizes, size_limits=(10, 100))
        _, _, paths = glyph.plot()
        assert np.array_equal(paths.get_array(), values), "Colour array must be the raw values"
        out = paths.get_sizes()
        assert np.array_equal(np.argsort(out), np.argsort(sizes)), "Sizes must follow sizes array"

    def test_sizes_none_regression(self, xy):
        """``sizes=None`` keeps the original scalar-size behaviour.

        Test scenario:
            Every marker uses the scalar ``point_size`` and no legend is made.
        """
        x, y = xy
        glyph = ScatterGlyph(x, y, point_size=42)
        _, _, paths = glyph.plot()
        out = paths.get_sizes()
        assert np.all(out == 42), f"All markers should use point_size 42, got {set(out)}"
        assert glyph.size_legend_artist is None, "No legend without sizes"

    def test_resolve_marker_area_scalar_without_sizes(self, xy):
        """``_resolve_marker_area`` returns the scalar ``point_size``.

        Test scenario:
            With no ``sizes`` the helper returns the scalar option, not an
            array.
        """
        x, y = xy
        glyph = ScatterGlyph(x, y, point_size=33)
        assert glyph._resolve_marker_area() == 33, "Should return scalar point_size"

    def test_resolve_marker_area_array_with_sizes(self, xy):
        """``_resolve_marker_area`` returns a per-point area array with sizes.

        Test scenario:
            With ``sizes`` the helper returns an array spanning ``size_limits``.
        """
        x, y = xy
        glyph = ScatterGlyph(x, y, sizes=np.arange(1.0, 6.0), size_limits=(10, 200))
        areas = glyph._resolve_marker_area()
        assert isinstance(areas, np.ndarray), "Should return a per-point array"
        assert np.isclose(areas.min(), 10.0) and np.isclose(areas.max(), 200.0), (
            f"Areas should span size_limits, got [{areas.min()}, {areas.max()}]"
        )

    def test_mismatched_sizes_raises(self, xy):
        """A ``sizes`` array not matching x/y length raises ``ValueError``.

        Test scenario:
            len(sizes) != len(x) is rejected at construction.
        """
        x, y = xy
        with pytest.raises(ValueError, match="sizes must match"):
            ScatterGlyph(x, y, sizes=np.array([1.0, 2.0]))
