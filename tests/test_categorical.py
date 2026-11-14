"""Tests for the shared categorical-glyph helpers in
`cleopatra.glyphs.stats._categorical`.

`resolve_categories` / `resolve_category_colors` are exercised end-to-end by the
`BarGlyph` and `PieGlyph` suites; this module adds the direct, boundary-level
coverage for the validation branches and for the multi-series helpers
(`resolve_series`, `resolve_series_colors`) that back stacked / grouped bars.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.colors import to_hex

from cleopatra.glyphs.stats._categorical import (
    resolve_categories,
    resolve_category_colors,
    resolve_series,
    resolve_series_colors,
)


class TestResolveCategories:
    """Direct boundary tests for the single-series validator."""

    def test_non_1d_values_raise(self):
        """A values array of rank != 1 is rejected, naming the rank it got.

        Test scenario:
            A 2-D block reaches `resolve_categories` only when a caller uses it
            directly (BarGlyph routes 2-D to `resolve_series`); it must still be
            refused with the "1-D" message.
        """
        with pytest.raises(ValueError, match=r"values must be 1-D, got 2-D") as exc:
            resolve_categories(["a", "b"], [[1.0], [2.0]])
        assert "2-D" in str(exc.value), f"message should name the rank, got: {exc.value}"

    def test_length_mismatch_raises(self):
        """Differing category / value counts raise, naming both counts.

        Test scenario:
            Two labels but three values must raise a "same length" error.
        """
        with pytest.raises(ValueError, match="same length"):
            resolve_categories(["a", "b"], [1.0, 2.0, 3.0])


class TestResolveSeries:
    """Validation and normalisation of a 2-D multi-series block."""

    def test_normalises_labels_and_shape(self):
        """A valid 2-D block yields string labels and a float array of its shape.

        Test scenario:
            A (2 categories, 2 series) block returns ['a', 'b'] and shape (2, 2).
        """
        labels, values = resolve_series(["a", "b"], [[1, 2], [3, 4]])
        assert labels == ["a", "b"], f"labels should be strings, got {labels}"
        assert values.shape == (2, 2), f"expected shape (2, 2), got {values.shape}"
        assert values.dtype == float, f"values should be float, got {values.dtype}"

    @pytest.mark.parametrize("values", [[1.0, 2.0], np.zeros((2, 2, 2))])
    def test_non_2d_values_raise(self, values):
        """A values array of rank != 2 is rejected as not multi-series.

        Args:
            values: A 1-D and a 3-D input, both outside the 2-D contract.

        Test scenario:
            Only a 2-D (n_categories, n_series) block is a multi-series input;
            anything else raises naming the required 2-D shape.
        """
        with pytest.raises(ValueError, match="must be 2-D"):
            resolve_series(["a", "b"], values)

    def test_row_count_must_match_categories(self):
        """A row count differing from the label count raises, naming both.

        Test scenario:
            Two labels against three value rows must raise a "number of value
            rows" error.
        """
        with pytest.raises(ValueError, match="number of value rows"):
            resolve_series(["a", "b"], [[1, 2], [3, 4], [5, 6]])

    def test_empty_raises(self):
        """An empty 2-D block (no categories) is rejected as empty.

        Test scenario:
            Zero categories with a (0, 3) array has matching row count but zero
            size, so it trips the non-empty guard rather than the shape guard.
        """
        with pytest.raises(ValueError, match="non-empty"):
            resolve_series([], np.zeros((0, 3)))

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_values_raise(self, bad):
        """NaN or inf anywhere in the block is rejected up front.

        Args:
            bad: The non-finite value seeded into one cell.

        Test scenario:
            A single non-finite cell must raise a "finite" error rather than be
            drawn silently.
        """
        with pytest.raises(ValueError, match="finite"):
            resolve_series(["a", "b"], [[1.0, 2.0], [3.0, bad]])


class TestResolveSeriesColors:
    """Per-series colour resolution (explicit list or colormap sampling)."""

    def test_explicit_colours_returned_verbatim(self):
        """An explicit per-series list of the right length is returned as given.

        Test scenario:
            Two colours for two series come back unchanged, in order.
        """
        out = resolve_series_colors(2, ["#ff0000", "#00ff00"], "viridis")
        assert out == ["#ff0000", "#00ff00"], f"colours should pass through, got {out}"

    def test_length_mismatch_raises(self):
        """An explicit colour list of the wrong length is refused.

        Test scenario:
            One colour for three series raises, naming the series count.
        """
        with pytest.raises(ValueError, match="number of series"):
            resolve_series_colors(3, ["#f00"], "viridis")

    def test_cmap_sampled_once_per_series(self):
        """Without explicit colours, the colormap is sampled once per series.

        Test scenario:
            Three series yield three distinct colours drawn from the colormap.
        """
        out = resolve_series_colors(3, None, "viridis")
        assert len(out) == 3, f"expected 3 colours, got {len(out)}"
        hexes = [to_hex(c) for c in out]
        assert len(set(hexes)) == 3, f"series colours should differ, got {hexes}"

    def test_single_series_samples_colormap_midpoint(self):
        """A lone series samples the colormap's midpoint, not an endpoint.

        Test scenario:
            With one series the sampler must use position 0.5, not the 0.0
            endpoint a plain ``np.linspace(0, 1, 1)`` would return.
        """
        out = resolve_series_colors(1, None, "viridis")
        assert len(out) == 1, f"expected 1 colour, got {len(out)}"
        viridis = mpl.colormaps["viridis"]
        assert to_hex(out[0]) == to_hex(viridis(0.5)), (
            f"a lone series should be the colormap midpoint, got {to_hex(out[0])}"
        )
        assert to_hex(out[0]) != to_hex(viridis(0.0)), (
            "a lone series must not collapse to the 0.0 endpoint"
        )


class TestResolveCategoryColors:
    """A regression guard for the single-series colour resolver's reuse here."""

    def test_palette_cycles_to_requested_length(self):
        """More categories than palette swatches cycle the palette.

        Test scenario:
            Requesting more colours than a palette has wraps around rather than
            raising.
        """
        out = resolve_category_colors(30, None, "category12")
        assert len(out) == 30, f"expected 30 colours, got {len(out)}"
