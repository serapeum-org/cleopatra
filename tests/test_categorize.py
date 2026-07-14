"""Tests for distinct-value categorical colouring -- issue #204.

Covers:

* `cleopatra.styles.categorize` (distinct-value -> colour mapping: sorted
  numeric/string categories, mixed-type fallback, null dropping, cmap
  cycling vs. continuous-cmap sampling, the all-null error path).
* `Glyph._prepare_categorical_mapping` / `Glyph.create_categorical_legend`
  and the `scheme="categorical"` short-circuit added to
  `Glyph._prepare_scalar_mapping`.
* Integration through `PolygonGlyph` and `ScatterGlyph` (distinct colours,
  a `disjoint_legend` instead of a colorbar, string- and integer-coded
  attributes, missing-value handling).
* The glyph-scope guard: only glyphs with `_SUPPORTS_CATEGORICAL_SCHEME`
  accept `scheme="categorical"`; the others (whose values are a continuous
  magnitude) raise a clear `ValueError` instead of mis-colouring.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

from cleopatra.flow_glyph import FlowGlyph
from cleopatra.glyph import CATEGORICAL_DEFAULT_CMAP, Glyph
from cleopatra.polygon_glyph import PolygonGlyph
from cleopatra.scatter_glyph import ScatterGlyph
from cleopatra.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS
from cleopatra.styles import categorize
from cleopatra.vector_glyph import VectorGlyph


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


def _make_options(**overrides) -> dict:
    """Build a Glyph ``default_options`` dict with auto colour limits.

    Args:
        **overrides: Option keys to override on top of the shared defaults.

    Returns:
        dict: A copy of ``STYLE_DEFAULTS`` with ``vmin``/``vmax`` unset plus
            any overrides applied.
    """
    opts = STYLE_DEFAULTS.copy()
    opts["vmin"] = None
    opts["vmax"] = None
    opts.update(overrides)
    return opts


class TestCategorize:
    """Tests for the public ``cleopatra.styles.categorize`` function."""

    def test_distinct_integer_codes_sorted(self):
        """Integer class codes become sorted, deduplicated categories.

        Test scenario:
            Repeated/unsorted integers collapse to their sorted unique set.
        """
        categories, palette = categorize([3, 1, 2, 1])
        assert list(categories) == [1, 2, 3], f"Unexpected categories: {categories}"
        assert len(palette) == 3, f"Expected 3 colours, got {len(palette)}"

    def test_string_categories_sorted_alphabetically(self):
        """String labels are sorted alphabetically.

        Test scenario:
            Land-cover-style string labels come back in alpha order.
        """
        categories, _ = categorize(["urban", "water", "forest"])
        assert list(categories) == [
            "forest",
            "urban",
            "water",
        ], f"Unexpected order: {categories}"

    def test_none_and_nan_dropped(self):
        """`None` and `NaN` entries never become their own category.

        Test scenario:
            A mix of valid labels, `None`, and `NaN` yields only the three
            valid, deduplicated categories.
        """
        categories, _ = categorize(["urban", "water", None, "forest", np.nan])
        assert list(categories) == [
            "forest",
            "urban",
            "water",
        ], f"Null entries leaked into categories: {categories}"

    def test_colors_aligned_with_categories(self):
        """Colours and categories have the same length, one-to-one.

        Test scenario:
            Five distinct values produce five colours.
        """
        categories, palette = categorize(list("abcde"))
        assert len(palette) == len(categories) == 5

    def test_cycles_past_cmap_size(self):
        """More categories than the cmap's colours cycles back to the start.

        Test scenario:
            `tab10` has 10 colours; 12 categories repeats colours 0 and 1.
        """
        categories, palette = categorize(list(range(12)), cmap="tab10")
        assert len(categories) == 12, f"Expected 12 categories, got {len(categories)}"
        assert palette[10] == palette[0], "Colour 10 should cycle back to colour 0"
        assert palette[11] == palette[1], "Colour 11 should cycle back to colour 1"

    def test_within_cmap_size_no_repeats(self):
        """Categories within the cmap's size get distinct colours.

        Test scenario:
            3 categories against `tab10` (10 colours) never repeat.
        """
        _, palette = categorize(["a", "b", "c"], cmap="tab10")
        assert len(set(palette)) == 3, f"Expected 3 distinct colours: {palette}"

    def test_continuous_cmap_is_sampled_not_cycled(self):
        """A continuous (non-`ListedColormap`) cmap is sampled, not cycled.

        Test scenario:
            `"coolwarm"` has no discrete `.colors` list (it is a
            `LinearSegmentedColormap`, unlike the perceptually-uniform maps
            such as `"viridis"`, which modern matplotlib implements as a
            256-entry `ListedColormap`); categorize samples it at N evenly
            spaced points, so distinct categories still get distinct colours
            even past what a 10-swatch qualitative map would offer.
        """
        assert getattr(matplotlib.colormaps["coolwarm"], "colors", None) is None, (
            "Test assumption: 'coolwarm' must have no discrete .colors list"
        )
        categories, palette = categorize(list(range(15)), cmap="coolwarm")
        assert len(categories) == 15
        assert len(set(palette)) == 15, "Sampled continuous cmap should not repeat"

    def test_mixed_unorderable_types_fall_back_to_first_seen_order(self):
        """Unorderable mixed types keep first-encounter order instead of raising.

        Test scenario:
            Mixing `int` and `str` values cannot be sorted in Python 3; the
            categories come back in the order first encountered rather than
            raising `TypeError`.
        """
        categories, _ = categorize([1, "a", 1, "b"])
        assert list(categories) == [1, "a", "b"], f"Unexpected order: {categories}"

    def test_returned_categories_hex_colors(self):
        """Colours are returned as hex strings.

        Test scenario:
            Every palette entry parses as a matplotlib colour and round-trips
            through `to_hex`.
        """
        _, palette = categorize(["x", "y"])
        for color in palette:
            assert color == mcolors.to_hex(color), f"Not a hex string: {color}"

    def test_all_null_raises(self):
        """An all-null input has no category to assign.

        Test scenario:
            Every entry is `None`/`NaN`; there is nothing to categorize.
        """
        with pytest.raises(ValueError, match="no non-null entries"):
            categorize([None, np.nan])

    def test_2d_input_is_flattened(self):
        """A 2-D values array is categorized on its flattened entries.

        Test scenario:
            A grid of two repeating labels yields the same two categories
            as the flat equivalent.
        """
        grid = np.array([["a", "b"], ["b", "a"]], dtype=object)
        categories, _ = categorize(grid)
        assert list(categories) == ["a", "b"], f"Unexpected categories: {categories}"

    def test_equal_but_differently_typed_values_collapse_to_one_category(self):
        """`1`, `1.0`, and `True` dedupe into a single category, as documented.

        Test scenario:
            Deduplication is by Python `hash`/`==` (the same rule any
            `dict`/`set` uses), so an integer code, its float equivalent, and
            the boolean `True` -- all equal to each other -- collapse into
            one category rather than three, matching the documented caveat.
        """
        categories, _ = categorize(np.array([1, True, 0, False, 1.0], dtype=object))
        assert list(categories) == [0, 1], f"Expected 2 categories, got {categories}"

    def test_ragged_non_hashable_entries_raise_type_error(self):
        """A ragged sequence of non-hashable entries raises `TypeError`.

        Test scenario:
            Differently-sized nested lists cannot be coerced into a
            rectangular array, so each list survives as its own (unhashable)
            element and `dict.fromkeys` raises `TypeError` -- matching the
            documented contract.
        """
        with pytest.raises(TypeError, match="unhashable"):
            categorize([[1, 2], [3, 4, 5], [1, 2]])

    def test_rectangular_nested_sequences_are_flattened_not_rejected(self):
        """Equal-length nested sequences are treated as a 2-D scalar grid.

        Test scenario:
            A rectangular list-of-lists is coerced to a 2-D array (like
            `test_2d_input_is_flattened`) and its scalar elements become the
            categories -- it does not raise, and the "list" values never
            become categories themselves.
        """
        categories, _ = categorize([[1, 2], [3, 4], [1, 2]])
        assert list(categories) == [1, 2, 3, 4], f"Unexpected categories: {categories}"


class TestGlyphPrepareCategoricalMapping:
    """Tests for ``Glyph._prepare_categorical_mapping`` and the routing."""

    def test_returns_norm_cbar_edges_triple(self):
        """The helper returns a BoundaryNorm, empty cbar_kw, and code edges.

        Test scenario:
            Three distinct values give a 3-colour BoundaryNorm with edges
            `[-0.5, 0.5, 1.5, 2.5]`.
        """
        g = PolygonGlyph([np.zeros((3, 2))] * 3, values=np.array(["a", "b", "c"]))
        norm, cbar_kw, edges = g._prepare_categorical_mapping(np.array(["a", "b", "c"]))
        assert isinstance(norm, mcolors.BoundaryNorm), f"Expected BoundaryNorm: {norm}"
        assert cbar_kw == {}, f"cbar_kw should be empty for categorical: {cbar_kw}"
        assert np.allclose(edges, [-0.5, 0.5, 1.5, 2.5]), f"Unexpected edges: {edges}"

    def test_populates_categorical_side_channel(self):
        """`self._categorical` carries codes/cmap/colors/labels.

        Test scenario:
            After the call, `self._categorical` has the four expected keys
            with per-element codes aligned to the input order.
        """
        g = PolygonGlyph([np.zeros((3, 2))] * 3, values=np.array(["b", "a", "b"]))
        g._prepare_categorical_mapping(np.array(["b", "a", "b"]))
        categorical = g._categorical
        assert set(categorical) == {"codes", "cmap", "colors", "labels"}
        assert np.allclose(
            categorical["codes"], [1.0, 0.0, 1.0]
        ), f"Unexpected codes: {categorical['codes']}"
        assert categorical["labels"] == ["a", "b"]
        assert isinstance(categorical["cmap"], mcolors.ListedColormap)

    def test_default_cmap_falls_back_to_qualitative_palette(self):
        """Leaving `cmap` unset resolves to a qualitative default, not `coolwarm_r`.

        Test scenario:
            With no explicit `cmap`, categories get `CATEGORICAL_DEFAULT_CMAP`
            (`"tab10"`) colours instead of samples from the glyph's continuous
            diverging default -- otherwise "distinct colours" would silently
            come from a red-white-blue gradient.
        """
        g = PolygonGlyph([np.zeros((3, 2))] * 3, values=np.array(["a", "b", "c"]))
        g._prepare_categorical_mapping(np.array(["a", "b", "c"]))
        expected = [
            mcolors.to_hex(c)
            for c in matplotlib.colormaps[CATEGORICAL_DEFAULT_CMAP].colors[:3]
        ]
        assert g._categorical["colors"] == expected, (
            f"Expected the {CATEGORICAL_DEFAULT_CMAP} palette, got "
            f"{g._categorical['colors']}"
        )

    def test_explicit_cmap_is_honoured_even_if_continuous(self):
        """An explicitly chosen `cmap` is never overridden by the fallback.

        Test scenario:
            Passing `cmap="coolwarm"` explicitly is still respected even
            though it is the same continuous colormap the glyph's own
            default resolves to (just spelled out instead of left implicit)
            -- the fallback only triggers when `cmap` was left at the
            glyph's own unmodified default value.
        """
        g = PolygonGlyph(
            [np.zeros((3, 2))] * 3, values=np.array(["a", "b", "c"]), cmap="coolwarm"
        )
        g._prepare_categorical_mapping(np.array(["a", "b", "c"]))
        assert len(set(g._categorical["colors"])) == 3, "Should still get 3 colours"

    def test_default_cmap_fallback_matches_a_colormap_object_too(self):
        """A `Colormap` instance equivalent to the default also falls back.

        Test scenario:
            Passing `cmap=matplotlib.colormaps["coolwarm_r"]` (an object,
            not the bare string) must be recognised as "still at the
            glyph's own default" and fall back to the qualitative palette,
            the same as leaving `cmap` unset entirely -- `Colormap` has no
            `__eq__`, so a naive `==` comparison would miss this and leave
            the diverging gradient sampled instead.
        """
        g = PolygonGlyph(
            [np.zeros((3, 2))] * 3,
            values=np.array(["a", "b", "c"]),
            cmap=matplotlib.colormaps["coolwarm_r"],
        )
        g._prepare_categorical_mapping(np.array(["a", "b", "c"]))
        expected = [
            mcolors.to_hex(c)
            for c in matplotlib.colormaps[CATEGORICAL_DEFAULT_CMAP].colors[:3]
        ]
        assert g._categorical["colors"] == expected, (
            f"Expected the {CATEGORICAL_DEFAULT_CMAP} fallback palette, got "
            f"{g._categorical['colors']}"
        )

    def test_prepare_scalar_mapping_routes_to_categorical(self):
        """`_prepare_scalar_mapping` short-circuits for `scheme="categorical"`.

        Test scenario:
            The shared entry point returns a BoundaryNorm and populates
            `self._categorical` when the scheme is `"categorical"`.
        """
        g = PolygonGlyph(
            [np.zeros((3, 2))] * 2,
            values=np.array(["x", "y"]),
            scheme="categorical",
        )
        norm, cbar_kw, _ = g._prepare_scalar_mapping(np.array(["x", "y"]))
        assert isinstance(norm, mcolors.BoundaryNorm)
        assert g._categorical is not None, "Categorical side-channel not populated"

    def test_unsupported_glyph_raises(self):
        """A base `Glyph` (no categorical support) raises a clear error.

        Test scenario:
            `_SUPPORTS_CATEGORICAL_SCHEME` is `False` on the base class, so
            calling the helper directly raises `ValueError`.
        """
        g = Glyph(default_options=_make_options(scheme="categorical"))
        with pytest.raises(ValueError, match="does not support scheme='categorical'"):
            g._prepare_categorical_mapping(np.array(["a", "b"]))

    def test_create_categorical_legend_before_prepare_raises(self):
        """Calling the legend builder before the mapping is prepared errors clearly.

        Test scenario:
            A freshly constructed glyph has `self._categorical is None`;
            `create_categorical_legend` must raise a clear `ValueError`
            rather than a raw `TypeError` from indexing `None`.
        """
        g = PolygonGlyph([np.zeros((3, 2))] * 2, values=np.array(["a", "b"]))
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match="before a scheme='categorical' mapping"):
            g.create_categorical_legend(ax)

    def test_warns_on_conflicting_color_scale(self):
        """`scheme="categorical"` with a non-linear `color_scale` warns.

        Test scenario:
            Categorical colouring owns the norm, so a conflicting
            `color_scale` is ignored -- and a warning says so.
        """
        glyph = ScatterGlyph(
            np.arange(4.0),
            np.zeros(4),
            values=np.array(["a", "b", "a", "b"]),
            scheme="categorical",
            color_scale="midpoint",
        )
        with pytest.warns(UserWarning, match="color_scale"):
            glyph.plot()

    def test_warns_on_conflicting_levels(self):
        """`scheme="categorical"` together with `levels` warns.

        Test scenario:
            The categorical mapping determines the classes, so `levels` is
            ignored -- and a warning says so.
        """
        glyph = ScatterGlyph(
            np.arange(4.0),
            np.zeros(4),
            values=np.array(["a", "b", "a", "b"]),
            scheme="categorical",
            levels=3,
        )
        with pytest.warns(UserWarning, match="levels"):
            glyph.plot()


class TestPolygonGlyphCategorical:
    """Integration tests for `scheme="categorical"` through PolygonGlyph."""

    @pytest.fixture()
    def polys_land_use(self):
        """Six triangles labelled with a repeating 3-class land-use attribute.

        Returns:
            tuple[list[np.ndarray], np.ndarray]: polygon vertices and string
                land-use labels.
        """
        polys = [np.array([[i, 0.0], [i + 1, 0.0], [i + 0.5, 1.0]]) for i in range(6)]
        labels = np.array(["forest", "water", "urban", "forest", "water", "forest"])
        return polys, labels

    def test_distinct_values_get_distinct_colors(self, polys_land_use):
        """Each unique value maps to its own fill colour.

        Test scenario:
            Three unique labels produce a 3-colour `ListedColormap`.
        """
        polys, labels = polys_land_use
        glyph = PolygonGlyph(polys, values=labels, scheme="categorical", cmap="tab10")
        _, _, pc = glyph.plot()
        assert isinstance(pc.cmap, mcolors.ListedColormap)
        assert len(pc.cmap.colors) == 3, f"Expected 3 colours: {pc.cmap.colors}"

    def test_disjoint_legend_not_colorbar(self, polys_land_use):
        """A categorical choropleth draws a legend, never a colorbar.

        Test scenario:
            `glyph.cbar` stays `None`; `glyph.category_legend` is created
            with one label per distinct value.
        """
        polys, labels = polys_land_use
        glyph = PolygonGlyph(polys, values=labels, scheme="categorical")
        glyph.plot()
        assert glyph.cbar is None, "A categorical choropleth must not draw a colorbar"
        assert glyph.category_legend is not None, "A disjoint legend should be drawn"
        legend_labels = {t.get_text() for t in glyph.category_legend.get_texts()}
        assert legend_labels == {
            "forest",
            "urban",
            "water",
        }, f"Unexpected legend labels: {legend_labels}"

    def test_category_legend_kwargs_overrides_title_and_placement(self, polys_land_use):
        """`category_legend_kwargs` is forwarded to the disjoint legend.

        Test scenario:
            An explicit `title` in `category_legend_kwargs` overrides the
            `cbar_label`-derived default, mirroring `size_legend_kwargs`.
        """
        polys, labels = polys_land_use
        glyph = PolygonGlyph(
            polys,
            values=labels,
            scheme="categorical",
            category_legend_kwargs={"title": "Land use", "loc": "lower left"},
        )
        glyph.plot()
        assert glyph.category_legend.get_title().get_text() == "Land use"

    def test_add_colorbar_false_suppresses_legend(self, polys_land_use):
        """`add_colorbar=False` also suppresses the categorical legend.

        Test scenario:
            The same toggle that suppresses a colorbar suppresses the
            disjoint legend for shared-axes composition.
        """
        polys, labels = polys_land_use
        glyph = PolygonGlyph(polys, values=labels, scheme="categorical")
        glyph.plot(add_colorbar=False)
        assert glyph.category_legend is None
        assert glyph.cbar is None

    def test_integer_coded_attribute(self, polys_land_use):
        """Integer class codes (e.g. D8 flow directions) work the same way.

        Test scenario:
            Small integer codes map to as many distinct colours as codes.
        """
        polys, _ = polys_land_use
        codes = np.array([1, 2, 4, 8, 1, 2])
        glyph = PolygonGlyph(polys, values=codes, scheme="categorical")
        _, _, pc = glyph.plot()
        assert len(pc.cmap.colors) == 4, f"Expected 4 colours: {pc.cmap.colors}"
        assert set(
            glyph.category_legend.get_texts()[i].get_text() for i in range(4)
        ) == {
            "1",
            "2",
            "4",
            "8",
        }

    def test_missing_value_renders_transparent(self, polys_land_use):
        """A polygon whose value is `None` gets a masked (transparent) code.

        Test scenario:
            One `None` entry among valid labels does not become its own
            category, and its per-polygon code is masked/NaN.
        """
        polys, _ = polys_land_use
        labels = np.array(
            ["forest", "water", None, "forest", "water", "forest"], dtype=object
        )
        glyph = PolygonGlyph(polys, values=labels, scheme="categorical")
        _, _, pc = glyph.plot()
        codes = np.ma.asarray(pc.get_array())
        assert codes.mask[2], "The None-valued polygon's code should be masked"
        assert len(pc.cmap.colors) == 2, "Only the 2 real labels become categories"

    def test_scheme_none_regression(self, polys_land_use):
        """Without `scheme`, numeric values still use the continuous path.

        Test scenario:
            Plain numeric values are unaffected by the categorical addition.
        """
        polys, _ = polys_land_use
        glyph = PolygonGlyph(polys, values=np.arange(6.0))
        _, _, pc = glyph.plot()
        assert not isinstance(pc.norm, mcolors.BoundaryNorm)
        assert glyph.category_legend is None

    def test_replot_from_categorical_to_continuous_clears_legend(self, polys_land_use):
        """Switching `scheme` off on a re-plot drops the stale legend.

        Test scenario:
            A glyph first plotted with `scheme="categorical"` (legend, no
            colorbar) is re-plotted with `scheme=None` and numeric values;
            the stale `category_legend` reference must not survive.
        """
        polys, labels = polys_land_use
        glyph = PolygonGlyph(polys, values=labels, scheme="categorical")
        glyph.plot()
        assert glyph.category_legend is not None and glyph.cbar is None

        glyph.default_options["scheme"] = None
        glyph.values = np.arange(6.0)
        glyph.plot()
        assert glyph.category_legend is None, "Stale legend must be cleared"
        assert glyph.cbar is not None, "The new continuous plot should draw a colorbar"

    def test_replot_from_continuous_to_categorical_clears_colorbar(self, polys_land_use):
        """Switching `scheme` on for a re-plot drops the stale colorbar.

        Test scenario:
            The reverse direction of the previous test: continuous first,
            categorical second.
        """
        polys, labels = polys_land_use
        glyph = PolygonGlyph(polys, values=np.arange(6.0))
        glyph.plot()
        assert glyph.cbar is not None and glyph.category_legend is None

        glyph.default_options["scheme"] = "categorical"
        glyph.values = labels
        glyph.plot()
        assert glyph.cbar is None, "Stale colorbar must be cleared"
        assert glyph.category_legend is not None, "The new categorical plot draws a legend"


class TestScatterGlyphCategorical:
    """Integration tests for `scheme="categorical"` through ScatterGlyph."""

    @pytest.fixture()
    def xy_species(self):
        """Six points labelled with a repeating 3-class species attribute.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: x, y, and species
                string labels.
        """
        x = np.arange(6.0)
        y = np.zeros_like(x)
        species = np.array(["oak", "pine", "oak", "birch", "pine", "oak"])
        return x, y, species

    def test_distinct_values_get_distinct_colors(self, xy_species):
        """Each unique species maps to its own point colour.

        Test scenario:
            Three unique labels produce a 3-colour `ListedColormap`.
        """
        x, y, species = xy_species
        glyph = ScatterGlyph(x, y, values=species, scheme="categorical")
        _, _, paths = glyph.plot()
        assert isinstance(paths.cmap, mcolors.ListedColormap)
        assert len(paths.cmap.colors) == 3

    def test_disjoint_legend_not_colorbar(self, xy_species):
        """A categorical scatter draws a legend, never a colorbar.

        Test scenario:
            `glyph.cbar` stays `None`; `glyph.category_legend` carries one
            label per distinct species.
        """
        x, y, species = xy_species
        glyph = ScatterGlyph(x, y, values=species, scheme="categorical")
        glyph.plot()
        assert glyph.cbar is None
        legend_labels = {t.get_text() for t in glyph.category_legend.get_texts()}
        assert legend_labels == {"oak", "pine", "birch"}

    def test_integer_coded_attribute(self, xy_species):
        """Integer class codes work the same way as string labels.

        Test scenario:
            Station-class integer codes map to distinct colours.
        """
        x, y, _ = xy_species
        codes = np.array([10, 20, 10, 30, 20, 10])
        glyph = ScatterGlyph(x, y, values=codes, scheme="categorical")
        _, _, paths = glyph.plot()
        assert len(paths.cmap.colors) == 3

    def test_missing_value_renders_transparent(self, xy_species):
        """A point whose value is `NaN` gets a masked (transparent) code.

        Test scenario:
            One `NaN` entry among valid labels does not become its own
            category.
        """
        x, y, _ = xy_species
        values = np.array(["oak", "pine", np.nan, "birch", "pine", "oak"], dtype=object)
        glyph = ScatterGlyph(x, y, values=values, scheme="categorical")
        _, _, paths = glyph.plot()
        codes = np.ma.asarray(paths.get_array())
        assert codes.mask[2], "The NaN-valued point's code should be masked"

    def test_scheme_none_regression(self, xy_species):
        """Without `scheme`, numeric values still use the continuous path.

        Test scenario:
            Plain numeric values are unaffected by the categorical addition.
        """
        x, y, _ = xy_species
        glyph = ScatterGlyph(x, y, values=np.arange(6.0))
        _, _, paths = glyph.plot()
        assert not isinstance(paths.norm, mcolors.BoundaryNorm)
        assert glyph.category_legend is None

    def test_replot_from_categorical_to_continuous_clears_legend(self, xy_species):
        """Switching `scheme` off on a re-plot drops the stale legend.

        Test scenario:
            A glyph first plotted with `scheme="categorical"` (legend, no
            colorbar) is re-plotted with `scheme=None` and numeric values;
            the stale `category_legend` reference must not survive.
        """
        x, y, species = xy_species
        glyph = ScatterGlyph(x, y, values=species, scheme="categorical")
        glyph.plot()
        assert glyph.category_legend is not None and glyph.cbar is None

        glyph.default_options["scheme"] = None
        glyph.values = np.arange(6.0)
        glyph.plot()
        assert glyph.category_legend is None, "Stale legend must be cleared"
        assert glyph.cbar is not None, "The new continuous plot should draw a colorbar"

    def test_size_legend_does_not_evict_category_legend(self, xy_species):
        """Combining `scheme="categorical"` with `size_legend` keeps both legends.

        Test scenario:
            `Axes.legend()` is single-slot per axes; `size_legend`'s internal
            call to it would otherwise silently replace the categorical
            legend drawn just before it. Both legends must remain actual
            children of the axes (not just non-`None` attributes) after
            `plot()`.
        """
        x, y, species = xy_species
        sizes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        glyph = ScatterGlyph(
            x, y, values=species, sizes=sizes, scheme="categorical", size_legend=True
        )
        _, ax, _ = glyph.plot()
        assert glyph.category_legend in ax.get_children(), (
            "The categorical legend must still be rendered, not silently evicted"
        )
        assert glyph.size_legend_artist in ax.get_children(), (
            "The size legend must also be rendered"
        )


class TestCategoricalSchemeGlyphScope:
    """Tests for which glyphs accept `scheme="categorical"`.

    `PolygonGlyph`/`ScatterGlyph` colour per-element nominal labels, so they
    support it. `VectorGlyph`/`FlowGlyph` colour a continuous magnitude
    (vector length / flow accumulation): they still accept the generic
    `scheme` option (for continuous classification), but must reject the
    `"categorical"` value specifically rather than silently mis-colouring.
    """

    @pytest.mark.parametrize("glyph_cls", [PolygonGlyph, ScatterGlyph])
    def test_supported_glyphs_flag_true(self, glyph_cls):
        """The supported glyphs declare `_SUPPORTS_CATEGORICAL_SCHEME`.

        Args:
            glyph_cls: A glyph class expected to support categorical values.

        Test scenario:
            The class attribute is `True`.
        """
        assert glyph_cls._SUPPORTS_CATEGORICAL_SCHEME is True

    def test_flow_glyph_rejects_categorical(self):
        """`FlowGlyph` rejects `scheme="categorical"` with a clear error.

        Test scenario:
            Magnitude-coloured flow lines cannot use nominal categories;
            plotting raises `ValueError` rather than mis-colouring silently.
        """
        paths = [np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[1.0, 0.0], [2.0, 1.0]])]
        glyph = FlowGlyph(paths, values=np.array([1.0, 5.0]), scheme="categorical")
        with pytest.raises(ValueError, match="does not support scheme='categorical'"):
            glyph.plot()

    def test_vector_glyph_rejects_categorical(self):
        """`VectorGlyph` rejects `scheme="categorical"` with a clear error.

        Test scenario:
            Same guard as `FlowGlyph`, exercised through `VectorGlyph`.
        """
        x, y = np.meshgrid(np.arange(4.0), np.arange(4.0))
        rng = np.random.default_rng(2)
        u = rng.uniform(0.1, 5.0, size=x.shape)
        v = rng.uniform(0.1, 5.0, size=x.shape)
        glyph = VectorGlyph(x, y, u, v, scheme="categorical")
        with pytest.raises(ValueError, match="does not support scheme='categorical'"):
            glyph.plot(kind="quiver")
