"""Tests for categorical (classified) colouring — issue #154.

Covers:

* `cleopatra.styling.styles.classify` and its helpers `_scheme_edges` /
  `_fisher_jenks_edges` (the numpy-only schemes incl. native Fisher-Jenks,
  explicit edges, the `BoundaryNorm` output, and the error paths).
* `Glyph._prepare_classified_mapping` and the `scheme` / `k` short-circuit
  added to `Glyph._prepare_scalar_mapping`.
* Integration through `ScatterGlyph` and `PolygonGlyph` (discrete classes,
  stepped colorbar, raw value preservation, and the `scheme=None` regression).
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

import cleopatra.styling.styles as styles_mod
from cleopatra.glyphs.base.glyph import Glyph
from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph, FacetLayout
from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
from cleopatra.glyphs.gridded.vector_glyph import VectorGlyph
from cleopatra.glyphs.primitives.flow_glyph import FlowGlyph
from cleopatra.glyphs.primitives.polygon_glyph import PolygonGlyph
from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph
from cleopatra.glyphs.stats.hexbin_glyph import HexbinGlyph
from cleopatra.glyphs.stats.kde_glyph import KDEGlyph
from cleopatra.styling.params import Classify, Contour
from cleopatra.styling.scaling import ColorScaling
from cleopatra.styling.styles import (
    CLASSIFY_OPTIONS,
    JENKS_SCHEMES,
    NUMPY_SCHEMES,
    _fisher_jenks_edges,
    _scheme_edges,
    classify,
)
from cleopatra.styling.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


@pytest.fixture()
def ramp():
    """A 0..99 linear ramp of 100 values.

    Returns:
        np.ndarray: ``np.arange(100.0)``, convenient because quantile and
            equal-interval edges coincide on a uniform ramp.
    """
    return np.arange(100.0)


def _make_options(**overrides) -> dict:
    """Build a Glyph ``default_options`` dict with auto colour limits.

    Args:
        **overrides: Option keys to override on top of the shared defaults.

    Returns:
        dict: A copy of ``STYLE_DEFAULTS`` with ``vmin`` / ``vmax`` unset
            (so limits resolve from the data) plus any overrides applied.
    """
    opts = STYLE_DEFAULTS.copy()
    opts["vmin"] = None
    opts["vmax"] = None
    opts.update(overrides)
    return opts


class TestModuleConstants:
    """Tests for the scheme-name constants and the shared option defaults."""

    def test_numpy_schemes_membership(self):
        """The numpy-only scheme tuple lists exactly the no-dependency schemes.

        Test scenario:
            The four numpy schemes are present and the Jenks names are not.
        """
        assert set(NUMPY_SCHEMES) == {
            "quantiles",
            "equal_interval",
            "percentiles",
            "std_mean",
        }, f"Unexpected numpy schemes: {NUMPY_SCHEMES}"
        assert "fisher_jenks" not in NUMPY_SCHEMES, "Jenks must not be numpy-only"

    def test_jenks_schemes_membership(self):
        """The Jenks tuple names the two optional-extra schemes.

        Test scenario:
            ``JENKS_SCHEMES`` is exactly the native Fisher-Jenks names.
        """
        assert set(JENKS_SCHEMES) == {
            "fisher_jenks",
            "natural_breaks",
        }, f"Unexpected Jenks schemes: {JENKS_SCHEMES}"

    def test_scheme_and_k_in_classify_options_not_shared_defaults(self):
        """``scheme`` / ``k`` live in CLASSIFY_OPTIONS, not the shared defaults.

        Test scenario:
            They default to ``None`` and ``5`` in CLASSIFY_OPTIONS, and are
            deliberately kept out of the shared DEFAULT_OPTIONS so glyphs that
            bypass the scalar-mapping pipeline reject `scheme` instead of
            silently ignoring it.
        """
        assert CLASSIFY_OPTIONS["scheme"] is None, "scheme should default to None"
        assert CLASSIFY_OPTIONS["k"] == 5, "k should default to 5"
        assert "scheme" not in STYLE_DEFAULTS, "scheme must not be in shared defaults"
        assert "k" not in STYLE_DEFAULTS, "k must not be in shared defaults"


class TestClassify:
    """Tests for the public ``cleopatra.styling.styles.classify`` function."""

    @pytest.mark.parametrize(
        "scheme, expected",
        [
            ("equal_interval", [0.0, 19.8, 39.6, 59.4, 79.2, 99.0]),
            ("quantiles", [0.0, 19.8, 39.6, 59.4, 79.2, 99.0]),
            ("percentiles", [0.0, 19.8, 39.6, 59.4, 79.2, 99.0]),
        ],
    )
    def test_uniform_ramp_edges(self, ramp, scheme, expected):
        """Count/width schemes give known edges on a uniform ramp.

        Args:
            ramp: The 0..99 ramp fixture.
            scheme: The scheme name under test.
            expected: The expected bin edges (k=5).

        Test scenario:
            On a uniform ramp, equal-interval and equal-count schemes all
            coincide on the same six edges.
        """
        edges, _ = classify(ramp, scheme, k=5)
        assert np.allclose(edges, expected), f"{scheme} edges {edges} != {expected}"

    def test_quantiles_equal_counts(self):
        """Quantile classes hold (near) equal counts of points.

        Test scenario:
            With 100 points and k=4, each of the four classes holds ~25
            points (digitize on the interior edges).
        """
        data = np.arange(100.0)
        edges, _ = classify(data, "quantiles", k=4)
        counts = np.bincount(np.clip(np.digitize(data, edges[1:-1]), 0, 3), minlength=4)
        assert set(counts) <= {24, 25, 26}, f"Quantile counts not balanced: {counts}"

    def test_equal_interval_equal_widths(self, ramp):
        """Equal-interval classes have uniform width.

        Test scenario:
            The differences between successive edges are all equal.
        """
        edges, _ = classify(ramp, "equal_interval", k=5)
        widths = np.diff(edges)
        assert np.allclose(widths, widths[0]), f"Widths not uniform: {widths}"

    def test_std_mean_breaks_and_ignores_k(self):
        """``std_mean`` builds mean±nσ breaks regardless of ``k``.

        Test scenario:
            Edges are [min, mean-σ, mean, mean+σ, max] for a symmetric ramp
            (the ±2σ breaks fall outside the data range and are dropped), and
            passing a different ``k`` does not change the result.
        """
        data = np.arange(100.0)
        edges_k5, _ = classify(data, "std_mean", k=5)
        edges_k9, _ = classify(data, "std_mean", k=9)
        assert np.allclose(edges_k5, edges_k9), "std_mean must ignore k"
        mean, std = data.mean(), data.std()
        assert np.isclose(edges_k5[0], data.min()), "first edge should be data min"
        assert np.isclose(edges_k5[-1], data.max()), "last edge should be data max"
        assert any(np.isclose(edges_k5, mean)), (
            f"mean break {mean} missing from {edges_k5}"
        )
        assert any(np.isclose(edges_k5, mean - std)), "mean-σ break missing"

    def test_explicit_edges_used_verbatim_sorted(self, ramp):
        """A non-string ``scheme`` is treated as explicit, sorted edges.

        Test scenario:
            An unsorted edge sequence is sorted ascending and used as-is;
            ``k`` is ignored.
        """
        edges, _ = classify(ramp, [50.0, 0.0, 99.0], k=5)
        assert np.allclose(edges, [0.0, 50.0, 99.0]), f"Edges not sorted: {edges}"

    def test_returns_boundary_norm_matching_edges(self, ramp):
        """The returned norm is a ``BoundaryNorm`` over the same edges.

        Test scenario:
            ``classify`` returns a ``(edges, BoundaryNorm)`` pair whose
            boundaries equal the edges.
        """
        edges, norm = classify(ramp, "equal_interval", k=5)
        assert isinstance(norm, mcolors.BoundaryNorm), (
            f"Expected BoundaryNorm, got {type(norm)}"
        )
        assert np.allclose(norm.boundaries, edges), "norm boundaries must equal edges"

    def test_case_insensitive_scheme_name(self, ramp):
        """Scheme names are matched case-insensitively.

        Test scenario:
            "Equal_Interval" resolves to the same edges as "equal_interval".
        """
        upper, _ = classify(ramp, "Equal_Interval", k=5)
        lower, _ = classify(ramp, "equal_interval", k=5)
        assert np.allclose(upper, lower), "Scheme lookup should be case-insensitive"

    def test_non_finite_values_ignored(self):
        """Non-finite entries do not influence the edges.

        Test scenario:
            A ramp with NaN/inf appended yields the same edges as the clean
            ramp, because non-finite values are filtered out first.
        """
        clean = np.arange(100.0)
        dirty = np.concatenate([clean, [np.nan, np.inf, -np.inf]])
        edges_clean, _ = classify(clean, "equal_interval", k=5)
        edges_dirty, _ = classify(dirty, "equal_interval", k=5)
        assert np.allclose(edges_clean, edges_dirty), (
            "Non-finite values must be ignored"
        )

    def test_duplicate_edges_collapsed(self):
        """Repeated quantile edges are de-duplicated to keep edges increasing.

        Test scenario:
            Heavily tied data makes interior quantiles coincide; the result
            still has strictly increasing edges (a valid BoundaryNorm).
        """
        data = np.array([0.0] * 90 + [1.0] * 10)
        edges, norm = classify(data, "quantiles", k=5)
        assert np.all(np.diff(edges) > 0), f"Edges must strictly increase: {edges}"
        assert isinstance(norm, mcolors.BoundaryNorm), (
            "Should still build a BoundaryNorm"
        )

    def test_no_finite_values_raises(self):
        """All-non-finite input raises a clear ``ValueError``.

        Test scenario:
            An all-NaN array cannot be classified.
        """
        with pytest.raises(ValueError, match="no finite entries"):
            classify(np.array([np.nan, np.inf]), "quantiles", k=5)

    def test_degenerate_no_spread_raises(self):
        """Constant data (no spread) raises a clear ``ValueError``.

        Test scenario:
            All-equal values collapse to a single edge, which cannot form a
            BoundaryNorm.
        """
        with pytest.raises(ValueError, match="no spread"):
            classify(np.full(10, 3.0), "quantiles", k=5)

    def test_unknown_scheme_name_raises(self, ramp):
        """An unrecognised scheme name raises ``ValueError`` listing valid ones.

        Test scenario:
            "rainbow" is not a scheme; the message names the valid schemes.
        """
        with pytest.raises(ValueError, match="Unknown classification scheme"):
            classify(ramp, "rainbow", k=5)

    @pytest.mark.parametrize("bad_k", [0, -1])
    def test_k_below_one_raises(self, ramp, bad_k):
        """``k < 1`` raises ``ValueError`` for the count/width schemes.

        Args:
            ramp: The ramp fixture.
            bad_k: An invalid class count.

        Test scenario:
            Fewer than one class is meaningless and rejected.
        """
        with pytest.raises(ValueError, match="`k` must be >= 1"):
            classify(ramp, "quantiles", k=bad_k)


class TestSchemeEdges:
    """Tests for the private ``_scheme_edges`` helper."""

    def test_routes_jenks_to_native_fisher_jenks(self):
        """A Jenks name is routed to the native Fisher-Jenks implementation.

        Test scenario:
            `fisher_jenks` returns `k + 1` edges computed in-process (no
            optional dependency), spanning the data range.
        """
        edges = _scheme_edges(np.arange(10.0), "fisher_jenks", k=3)
        assert len(edges) == 4, f"k=3 should give 4 edges, got {len(edges)}"
        assert edges[0] == 0.0 and edges[-1] == 9.0, f"Edges should span data: {edges}"

    def test_std_mean_k_ignored_branch(self):
        """``_scheme_edges`` does not validate ``k`` for ``std_mean``.

        Test scenario:
            ``k=0`` would raise for quantiles, but std_mean ignores k and
            returns edges without error.
        """
        edges = _scheme_edges(np.arange(100.0), "std_mean", k=0)
        assert edges[0] == 0.0 and edges[-1] == 99.0, (
            f"Unexpected std_mean edges: {edges}"
        )


class TestFisherJenksEdges:
    """Tests for the native ``_fisher_jenks_edges`` helper (no dependency)."""

    @staticmethod
    def _partition_sse(data, edges):
        """Sum of within-class squared deviations for the given bin edges."""
        data = np.sort(np.asarray(data, dtype=float))
        cls = np.searchsorted(edges[1:-1], data, side="left")
        return sum(
            ((data[cls == c] - data[cls == c].mean()) ** 2).sum()
            for c in np.unique(cls)
        )

    def test_matches_brute_force_optimum(self):
        """The DP finds the globally optimal (minimum-SSE) partition.

        Test scenario:
            For small samples, the within-class SSE achieved by the
            Fisher-Jenks edges equals the brute-force optimum over all
            contiguous k-partitions.
        """
        import itertools

        rng = np.random.default_rng(0)
        for _ in range(5):
            data = np.sort(rng.normal(size=11))
            for k in (2, 3, 4):
                best = min(
                    sum(
                        ((seg - seg.mean()) ** 2).sum()
                        for seg in np.split(data, list(cuts))
                        if seg.size
                    )
                    for cuts in itertools.combinations(range(1, data.size), k - 1)
                )
                edges = _fisher_jenks_edges(data, k)
                got = self._partition_sse(data, edges)
                assert np.isclose(got, best), f"k={k}: DP SSE {got} != optimum {best}"

    def test_classic_break(self):
        """A clear outlier is split into its own class.

        Test scenario:
            [1, 2, 3, 4, 5, 100] with k=2 breaks between 5 and 100.
        """
        edges = _fisher_jenks_edges(np.array([1.0, 2, 3, 4, 5, 100]), k=2)
        assert np.allclose(edges, [1.0, 5.0, 100.0]), f"Unexpected edges: {edges}"

    def test_edges_count_and_span(self):
        """`k` classes yield `k + 1` increasing edges spanning the data.

        Test scenario:
            k=5 on a ramp gives six strictly increasing edges from min to max.
        """
        edges = _fisher_jenks_edges(np.arange(50.0), k=5)
        assert len(edges) == 6, f"Expected 6 edges, got {len(edges)}"
        assert edges[0] == 0.0 and edges[-1] == 49.0, "Edges should span the data"
        assert np.all(np.diff(edges) > 0), f"Edges must be increasing: {edges}"

    def test_k_at_least_n_one_class_per_point(self):
        """When k >= number of points, every value becomes its own break.

        Test scenario:
            Three points with k=10 return the sorted values as edges.
        """
        edges = _fisher_jenks_edges(np.array([3.0, 1.0, 2.0]), k=10)
        assert np.allclose(edges, [1.0, 1.0, 2.0, 3.0]), f"Unexpected edges: {edges}"

    def test_natural_breaks_is_fisher_jenks_alias(self):
        """`natural_breaks` produces identical edges to `fisher_jenks`.

        Test scenario:
            Both scheme names route to the same exact optimisation.
        """
        data = np.arange(40.0)
        fisher, _ = classify(data, "fisher_jenks", k=4)
        natural, _ = classify(data, "natural_breaks", k=4)
        assert np.allclose(fisher, natural), "natural_breaks should equal fisher_jenks"

    def test_no_mapclassify_import(self):
        """Classifying with a Jenks scheme imports no optional dependency.

        Test scenario:
            After a fisher_jenks classification, `mapclassify` is absent from
            `sys.modules` — the algorithm is pure numpy.
        """
        import sys

        classify(np.arange(20.0), "fisher_jenks", k=4)
        assert "mapclassify" not in sys.modules, "Jenks must not import mapclassify"

    @pytest.mark.parametrize("scheme", ["fisher_jenks", "natural_breaks"])
    @pytest.mark.parametrize("bad_k", [0, -1])
    def test_jenks_k_below_one_raises(self, scheme, bad_k):
        """`k < 1` raises for the Jenks-family schemes too.

        Args:
            scheme: The Jenks scheme name under test.
            bad_k: An invalid class count.

        Test scenario:
            Fewer than one class is rejected before running the optimisation.
        """
        with pytest.raises(ValueError, match="`k` must be >= 1"):
            classify(np.arange(20.0), scheme, k=bad_k)

    def test_centering_preserves_optimum_and_shift_invariance(self):
        """Mean-centring keeps the breaks optimal and shift-invariant (L1).

        Test scenario:
            The same data offset by a large constant (1e9) yields the same
            partition — the breaks differ only by that constant — proving the
            centred SSE does not lose precision for extreme magnitudes.
        """
        data = np.array([0.0, 1, 2, 3, 4, 50, 51, 52, 200, 201])
        base, _ = classify(data, "fisher_jenks", k=3)
        shifted, _ = classify(data + 1e9, "fisher_jenks", k=3)
        assert np.allclose(shifted - 1e9, base), (
            f"Breaks should be shift-invariant; {shifted - 1e9} != {base}"
        )

    def test_large_input_samples_and_warns(self, monkeypatch):
        """Above `MAX_JENKS_N` the DP runs on a quantile sample and warns (M1).

        Args:
            monkeypatch: Lowers the cap so the sampling path is exercised
                without generating a huge array.

        Test scenario:
            Classifying more points than the cap emits a UserWarning and still
            returns `k + 1` strictly-increasing edges spanning the data range.
        """
        monkeypatch.setattr(styles_mod, "MAX_JENKS_N", 100)
        rng = np.random.default_rng(5)
        data = rng.normal(size=400)
        with pytest.warns(UserWarning, match="quantile sample"):
            edges, _ = classify(data, "fisher_jenks", k=5)
        assert len(edges) == 6, f"k=5 should give 6 edges, got {len(edges)}"
        assert np.all(np.diff(edges) > 0), f"Edges must be increasing: {edges}"
        assert np.isclose(edges[0], data.min()) and np.isclose(edges[-1], data.max()), (
            "Sampled breaks should still span the full data range"
        )

    def test_below_cap_does_not_warn(self, monkeypatch, recwarn):
        """At or below `MAX_JENKS_N` no sampling warning is emitted (M1).

        Args:
            monkeypatch: Sets a cap above the input size.
            recwarn: Captures any warnings raised.

        Test scenario:
            A modest input classified exactly (no sampling) warns nothing.
        """
        monkeypatch.setattr(styles_mod, "MAX_JENKS_N", 5000)
        classify(np.arange(50.0), "fisher_jenks", k=4)
        sampling = [w for w in recwarn.list if "quantile sample" in str(w.message)]
        assert not sampling, f"No sampling warning expected, got {sampling}"

    def test_ties_yield_fewer_classes(self):
        """Insufficient distinct values yield fewer than `k` classes (N1).

        Test scenario:
            Data with only two distinct values and k=4 collapses coincident
            breaks, returning strictly-increasing edges with < k + 1 entries.
        """
        edges, _ = classify(np.array([1.0, 1, 1, 2, 2]), "fisher_jenks", k=4)
        assert len(edges) < 5, f"Tied data should give < k+1 edges, got {len(edges)}"
        assert np.all(np.diff(edges) > 0), f"Edges must stay increasing: {edges}"


class TestGlyphPrepareClassifiedMapping:
    """Tests for ``Glyph._prepare_classified_mapping`` and the routing."""

    def test_returns_norm_cbar_edges_triple(self):
        """The helper returns the ``(norm, cbar_kw, edges)`` contract.

        Test scenario:
            A quantile scheme yields a BoundaryNorm, boundary ticks, and the
            bin edges in the ticks slot.
        """
        g = Glyph(default_options=_make_options(scheme="quantiles", k=4))
        norm, cbar_kw, edges = g._prepare_classified_mapping(
            np.arange(100.0), "quantiles"
        )
        assert isinstance(norm, mcolors.BoundaryNorm), "norm must be a BoundaryNorm"
        assert np.allclose(cbar_kw["ticks"], edges), "cbar ticks must be the edges"
        assert len(edges) == 5, f"k=4 should give 5 edges, got {len(edges)}"

    def test_extend_defaults_to_neither(self):
        """``extend`` defaults to ``'neither'`` when unset.

        Test scenario:
            No ``extend`` option present -> the colorbar does not extend.
        """
        g = Glyph(default_options=_make_options(scheme="quantiles", k=4))
        _, cbar_kw, _ = g._prepare_classified_mapping(np.arange(100.0), "quantiles")
        assert cbar_kw["extend"] == "neither", (
            f"extend should be 'neither', got {cbar_kw['extend']}"
        )

    def test_explicit_extend_is_honoured(self):
        """An explicit ``extend`` option is forwarded unchanged.

        Test scenario:
            ``extend='both'`` survives into the colorbar kwargs.
        """
        opts = _make_options(scheme="quantiles", k=4)
        opts["extend"] = "both"
        g = Glyph(default_options=opts)
        _, cbar_kw, _ = g._prepare_classified_mapping(np.arange(100.0), "quantiles")
        assert cbar_kw["extend"] == "both", "Explicit extend must be honoured"

    def test_prepare_scalar_mapping_routes_to_classified(self):
        """``_prepare_scalar_mapping`` short-circuits when ``scheme`` is set.

        Test scenario:
            With a scheme configured, the shared entry point returns a
            BoundaryNorm rather than the continuous (None) linear norm.
        """
        g = Glyph(default_options=_make_options(scheme="equal_interval", k=5))
        norm, _, edges = g._prepare_scalar_mapping(np.arange(100.0))
        assert isinstance(norm, mcolors.BoundaryNorm), (
            "scheme must route to BoundaryNorm"
        )
        assert len(edges) == 6, f"k=5 should give 6 edges, got {len(edges)}"

    def test_prepare_scalar_mapping_unchanged_without_scheme(self):
        """Without a scheme, the continuous linear path is unchanged.

        Test scenario:
            ``scheme=None`` keeps the default linear norm (``None``).
        """
        g = Glyph(default_options=_make_options())
        norm, _, _ = g._prepare_scalar_mapping(np.arange(100.0))
        assert norm is None, "Linear default should yield norm=None when no scheme"

    def test_k_option_controls_class_count(self):
        """The ``k`` option drives the number of classes.

        Test scenario:
            k=3 yields four edges (three classes).
        """
        g = Glyph(default_options=_make_options(scheme="quantiles", k=3))
        _, _, edges = g._prepare_scalar_mapping(np.arange(100.0))
        assert len(edges) == 4, f"k=3 should give 4 edges, got {len(edges)}"


class TestScatterGlyphScheme:
    """Integration tests for ``scheme`` through ScatterGlyph."""

    @pytest.fixture()
    def xy_values(self):
        """Ten points on a line with a 0..9 value ramp.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: x, y, and values.
        """
        x = np.arange(10.0)
        return x, np.zeros_like(x), x.copy()

    def test_scheme_produces_boundary_norm(self, xy_values):
        """A scheme colours the scatter through a BoundaryNorm.

        Test scenario:
            ``scheme='quantiles'`` makes the PathCollection use a discrete
            BoundaryNorm with k+1 boundaries.
        """
        x, y, v = xy_values
        glyph = ScatterGlyph(x, y, values=v)
        _, _, paths = glyph.plot(classify=Classify(scheme="quantiles", k=5))
        assert isinstance(paths.norm, mcolors.BoundaryNorm), (
            "scheme should set a BoundaryNorm"
        )
        assert len(paths.norm.boundaries) == 6, "k=5 should give 6 boundaries"

    def test_raw_values_preserved(self, xy_values):
        """Classification does not alter the underlying value array.

        Test scenario:
            ``get_array`` still returns the raw per-point values.
        """
        x, y, v = xy_values
        glyph = ScatterGlyph(x, y, values=v)
        _, _, paths = glyph.plot(classify=Classify(scheme="equal_interval", k=4))
        assert np.array_equal(paths.get_array(), v), "Raw values must be preserved"

    def test_colorbar_drawn_by_default(self, xy_values):
        """A classified scatter draws a (stepped) colorbar by default.

        Test scenario:
            ``add_colorbar`` defaults to True, so a colorbar is created.
        """
        x, y, v = xy_values
        glyph = ScatterGlyph(x, y, values=v)
        glyph.plot(classify=Classify(scheme="quantiles", k=5))
        assert glyph.cbar is not None, "A colorbar should be drawn by default"

    def test_add_colorbar_false_suppresses(self, xy_values):
        """``add_colorbar=False`` suppresses the colorbar with a scheme set.

        Test scenario:
            The plot-time override wins and no colorbar is created.
        """
        x, y, v = xy_values
        glyph = ScatterGlyph(x, y, values=v)
        glyph.plot(classify=Classify(scheme="quantiles", k=5), add_colorbar=False)
        assert glyph.cbar is None, "add_colorbar=False should suppress the colorbar"

    def test_scheme_none_regression(self, xy_values):
        """``scheme=None`` keeps the continuous (non-BoundaryNorm) behaviour.

        Test scenario:
            Without a scheme the scatter is not normalised by a BoundaryNorm.
        """
        x, y, v = xy_values
        glyph = ScatterGlyph(x, y, values=v)
        _, _, paths = glyph.plot()
        assert not isinstance(paths.norm, mcolors.BoundaryNorm), (
            "No scheme -> no BoundaryNorm"
        )


class TestPolygonGlyphScheme:
    """Integration tests for ``scheme`` through PolygonGlyph (choropleth)."""

    @pytest.fixture()
    def polys_values(self):
        """Ten triangles with a 0..9 value ramp.

        Returns:
            tuple[list[np.ndarray], np.ndarray]: polygon vertices and values.
        """
        polys = [np.array([[i, 0.0], [i + 1, 0.0], [i + 0.5, 1.0]]) for i in range(10)]
        return polys, np.arange(10.0)

    def test_five_discrete_classes(self, polys_values):
        """A k=5 quantile choropleth yields five discrete classes.

        Test scenario:
            The PolyCollection uses a BoundaryNorm whose six boundaries
            delimit five fill classes.
        """
        polys, values = polys_values
        glyph = PolygonGlyph(polys, values=values)
        _, _, pc = glyph.plot(classify=Classify(scheme="quantiles", k=5))
        assert isinstance(pc.norm, mcolors.BoundaryNorm), (
            "choropleth should use a BoundaryNorm"
        )
        assert len(pc.norm.boundaries) - 1 == 5, "Six boundaries delimit five classes"

    def test_raw_values_preserved(self, polys_values):
        """The polygon array still carries the raw values.

        Test scenario:
            ``get_array`` returns the unbinned per-polygon values.
        """
        polys, values = polys_values
        glyph = PolygonGlyph(polys, values=values)
        _, _, pc = glyph.plot(classify=Classify(scheme="quantiles", k=5))
        assert np.array_equal(pc.get_array(), values), "Raw values must be preserved"

    def test_discrete_colorbar_drawn(self, polys_values):
        """A discrete colorbar is attached for a classified choropleth.

        Test scenario:
            The colorbar exists and its norm is the discrete BoundaryNorm.
        """
        polys, values = polys_values
        glyph = PolygonGlyph(polys, values=values)
        glyph.plot(classify=Classify(scheme="quantiles", k=5))
        assert glyph.cbar is not None, "A discrete colorbar should be drawn"
        assert isinstance(glyph.cbar.norm, mcolors.BoundaryNorm), (
            "Colorbar norm should be discrete"
        )

    def test_scheme_none_regression(self, polys_values):
        """``scheme=None`` keeps the continuous choropleth behaviour.

        Test scenario:
            Without a scheme the PolyCollection is not BoundaryNorm-normalised.
        """
        polys, values = polys_values
        glyph = PolygonGlyph(polys, values=values)
        _, _, pc = glyph.plot()
        assert not isinstance(pc.norm, mcolors.BoundaryNorm), (
            "No scheme -> no BoundaryNorm"
        )


class TestArrayGlyphScheme:
    """Integration tests for `classify` through ArrayGlyph (classified raster, #351)."""

    @pytest.fixture()
    def ramp(self):
        """A 10x10 raster carrying a 0..99 value ramp.

        Returns:
            np.ndarray: The 2-D array to classify.
        """
        return np.arange(100.0).reshape(10, 10)

    def test_five_discrete_classes(self, ramp):
        """A k=5 quantile raster yields five discrete classes.

        Test scenario:
            The image uses a `BoundaryNorm` whose six boundaries delimit five
            fill classes.
        """
        glyph = ArrayGlyph(ramp)
        glyph.plot(classify=Classify(scheme="quantiles", k=5))
        assert isinstance(glyph.im.norm, mcolors.BoundaryNorm), (
            "raster should use a BoundaryNorm"
        )
        assert len(glyph.im.norm.boundaries) - 1 == 5, (
            "six boundaries delimit five classes"
        )

    def test_raw_values_preserved(self, ramp):
        """Classification does not alter the raster's value array.

        Test scenario:
            `get_array` still returns the raw cell values (max 99), not class
            codes.
        """
        glyph = ArrayGlyph(ramp)
        glyph.plot(classify=Classify(scheme="equal_interval", k=4))
        assert float(np.max(glyph.im.get_array())) == 99.0, (
            "raw values must be preserved on the mappable"
        )

    def test_stepped_colorbar_ticks_are_class_edges(self, ramp):
        """The classified colorbar steps on the class edges.

        Test scenario:
            A colorbar is drawn by default and its discrete norm's boundaries
            match `styles.classify` on the same data.
        """
        glyph = ArrayGlyph(ramp)
        glyph.plot(classify=Classify(scheme="quantiles", k=4))
        assert glyph.cbar is not None, "a stepped colorbar should be drawn by default"
        edges, _ = classify(ramp, "quantiles", k=4)
        assert np.allclose(glyph.cbar.norm.boundaries, edges), (
            "colorbar boundaries should be the class edges"
        )

    def test_explicit_edges_used_verbatim(self, ramp):
        """An explicit edge sequence is used as the class boundaries.

        Test scenario:
            `Classify(scheme=[0, 10, 50, 100])` gives exactly those boundaries.
        """
        glyph = ArrayGlyph(ramp)
        glyph.plot(classify=Classify(scheme=[0.0, 10.0, 50.0, 100.0]))
        assert list(glyph.im.norm.boundaries) == [0.0, 10.0, 50.0, 100.0], (
            "explicit edges should be used verbatim"
        )

    def test_natural_breaks_scheme(self, ramp):
        """The Jenks family works on a raster with numpy alone.

        Test scenario:
            `scheme="natural_breaks"` bins the field into k classes without
            importing mapclassify.
        """
        glyph = ArrayGlyph(ramp)
        glyph.plot(classify=Classify(scheme="natural_breaks", k=6))
        assert len(glyph.im.norm.boundaries) - 1 == 6, (
            "natural_breaks should give six classes"
        )

    def test_add_colorbar_false_suppresses(self, ramp):
        """`add_colorbar=False` suppresses the colorbar with a scheme set.

        Test scenario:
            The plot-time override wins and no colorbar is created.
        """
        glyph = ArrayGlyph(ramp)
        glyph.plot(classify=Classify(scheme="quantiles", k=5), add_colorbar=False)
        assert glyph.cbar is None, "add_colorbar=False should suppress the colorbar"

    def test_scheme_none_regression(self, ramp):
        """`scheme=None` keeps the continuous (non-BoundaryNorm) behaviour.

        Test scenario:
            A plain `plot()` leaves the image on a continuous norm.
        """
        glyph = ArrayGlyph(ramp)
        glyph.plot()
        assert not isinstance(glyph.im.norm, mcolors.BoundaryNorm), (
            "No scheme -> no BoundaryNorm"
        )

    def test_contourf_classified(self, ramp):
        """`kind="contourf"` draws filled bands at the class edges.

        Test scenario:
            A classified contourf renders and uses the discrete norm.
        """
        glyph = ArrayGlyph(ramp)
        _, ax = glyph.plot(
            kind="contourf", classify=Classify(scheme="equal_interval", k=5)
        )
        assert len(ax.collections) > 0, "contourf should draw filled bands"
        assert isinstance(glyph.im.norm, mcolors.BoundaryNorm), (
            "classified contourf should use a BoundaryNorm"
        )

    def test_categorical_rejected(self, ramp):
        """`scheme="categorical"` is rejected for a raster.

        Test scenario:
            A raster's cells are a continuous field, not nominal labels, so a
            categorical scheme raises.
        """
        glyph = ArrayGlyph(ramp)
        with pytest.raises(ValueError, match="categorical"):
            glyph.plot(classify=Classify(scheme="categorical"))

    def test_conflict_warning_attributed_to_caller(self, ramp):
        """The scheme/scale conflict warning points at the caller, not internals.

        Test scenario:
            A classified `ArrayGlyph.plot` with a conflicting `color_scale`
            warns once, and the warning's filename is this test module (the
            caller), not `array_glyph.py` -- the raster path resolves the norm
            at `plot` depth, so the shared `stacklevel` attributes correctly.
        """
        glyph = ArrayGlyph(ramp + 1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            glyph.plot(
                classify=Classify(scheme="quantiles", k=4), color=ColorScaling.log()
            )
        conflict = [w for w in caught if "color_scale" in str(w.message)]
        assert len(conflict) == 1, (
            f"expected exactly one conflict warning, got {caught}"
        )
        assert conflict[0].filename == __file__, (
            f"warning attributed to {conflict[0].filename}, not the caller {__file__}"
        )

    def test_bad_scheme_rolls_back(self, ramp):
        """An unknown scheme leaves no half-applied option on the glyph.

        Test scenario:
            A failed classified plot restores `scheme` to `None`, so a later
            plain `plot()` succeeds.
        """
        glyph = ArrayGlyph(ramp)
        with pytest.raises(ValueError):
            glyph.plot(classify=Classify(scheme="rainbow"))
        assert glyph.default_options.get("scheme") is None, (
            "a failed classified plot must not leave scheme set"
        )
        glyph.plot()

    def test_facet_shares_classes_over_stack(self):
        """Facet resolves one set of classes over the whole stack.

        Test scenario:
            Three slices with very different ranges share identical class edges
            spanning the entire stack, not per-panel bins.
        """
        stack = np.stack(
            [
                np.arange(100.0).reshape(10, 10),
                np.arange(1000.0, 1100.0).reshape(10, 10),
                np.arange(50.0, 150.0).reshape(10, 10),
            ]
        )
        grid = ArrayGlyph(stack).facet(
            FacetLayout(col="time"), classify=Classify(scheme="quantiles", k=4)
        )
        norms = [
            ax.get_images()[0].norm.boundaries.tolist()
            for row in grid.axes
            for ax in np.atleast_1d(row)
            if ax.get_images()
        ]
        assert len(norms) == 3, "three panels should be drawn"
        assert all(n == norms[0] for n in norms), "all panels share one set of classes"
        assert norms[0][0] == 0.0 and norms[0][-1] == 1099.0, (
            "class edges should span the whole stack"
        )

    def test_classified_raster_ignores_nan(self):
        """A classified raster bins only its finite cells, ignoring NaN.

        Test scenario:
            A ramp with a NaN cell classifies over the finite values (the class
            edges match `styles.classify` on the finite cells) and still renders.
        """
        arr = np.arange(100.0).reshape(10, 10)
        arr[0, 0] = np.nan
        glyph = ArrayGlyph(arr)
        glyph.plot(classify=Classify(scheme="quantiles", k=4))
        edges, _ = classify(arr[np.isfinite(arr)], "quantiles", k=4)
        assert isinstance(glyph.im.norm, mcolors.BoundaryNorm), "should classify"
        assert np.allclose(glyph.im.norm.boundaries, edges), (
            "NaN cells must be dropped before binning"
        )

    def test_animate_bad_scheme_rolls_back(self):
        """A bad scheme on `animate` leaves no half-applied option.

        Test scenario:
            A failed classified animation restores `scheme` to `None`, so a
            later plain `animate` succeeds.
        """
        stack = np.stack(
            [np.arange(100.0).reshape(10, 10), np.arange(100.0, 200.0).reshape(10, 10)]
        )
        glyph = ArrayGlyph(stack)
        with pytest.raises(ValueError):
            glyph.animate(["t0", "t1"], classify=Classify(scheme="rainbow"))
        assert glyph.default_options.get("scheme") is None, (
            "a failed classified animation must not leave scheme set"
        )
        glyph.animate(["t0", "t1"])

    def test_facet_bad_scheme_raises_before_figure(self, monkeypatch):
        """A raising facet scheme fails before the figure is created (no leak).

        Test scenario:
            A spreadless (constant) stack with a named scheme makes the
            shared-edge resolution raise; because that runs before `_facet_axes`,
            no figure is created (verified by asserting `_facet_axes` is never
            called).
        """
        stack = np.full((3, 4, 4), 5.0)
        glyph = ArrayGlyph(stack)
        calls = []
        monkeypatch.setattr(glyph, "_facet_axes", lambda *a, **k: calls.append(1))
        with pytest.raises(ValueError, match="spread"):
            glyph.facet(FacetLayout(col="time"), classify=Classify(scheme="quantiles"))
        assert not calls, "edge resolution must fail before the figure is created"

    def test_animate_shares_classes_over_frames(self):
        """Animate resolves one set of classes over the whole stack.

        Test scenario:
            The animated mappable's discrete norm spans every frame's data.
        """
        stack = np.stack(
            [
                np.arange(100.0).reshape(10, 10),
                np.arange(1000.0, 1100.0).reshape(10, 10),
            ]
        )
        glyph = ArrayGlyph(stack)
        glyph.animate(["t0", "t1"], classify=Classify(scheme="equal_interval", k=5))
        assert isinstance(glyph.im.norm, mcolors.BoundaryNorm), (
            "animation should classify through a BoundaryNorm"
        )
        assert float(glyph.im.norm.boundaries[0]) == 0.0, "edges start at the stack min"
        assert float(glyph.im.norm.boundaries[-1]) == 1099.0, (
            "edges end at the stack max"
        )


class TestSchemeGlyphScope:
    """Tests for which glyphs accept `scheme` (the M1 fix, extended for #351).

    `scheme`/`k` live in `CLASSIFY_OPTIONS` and are mixed into glyphs that
    classify their data: those whose colour mapping routes through
    `Glyph._prepare_scalar_mapping`, plus `ArrayGlyph`, which bypasses that
    pipeline but wires `scheme` into its own raster norm path
    (`_norm_cbar_and_ticks`). A glyph that neither routes through the pipeline
    nor wires it in — `MeshGlyph`, and `KDEGlyph` (whose `contourf` has its
    own `levels` discretisation) — must reject `scheme` rather than silently
    ignore it.
    """

    @pytest.mark.parametrize(
        "glyph_cls",
        [ScatterGlyph, PolygonGlyph, VectorGlyph, FlowGlyph, HexbinGlyph, ArrayGlyph],
    )
    def test_pipeline_glyphs_accept_scheme(self, glyph_cls):
        """Classifying glyphs expose `scheme`/`k` as accepted options.

        Args:
            glyph_cls: A glyph class that colours by class (the pipeline glyphs
                and, since #351, `ArrayGlyph`).

        Test scenario:
            `scheme` and `k` are in the class's option keys.
        """
        keys = glyph_cls.option_keys()
        assert "scheme" in keys and "k" in keys, (
            f"{glyph_cls.__name__} should accept scheme/k"
        )

    def test_array_glyph_accepts_scheme(self):
        """`ArrayGlyph` classifies its raster instead of rejecting `scheme` (#351).

        Test scenario:
            `ArrayGlyph(arr).plot(classify=Classify(scheme="quantiles", k=4))`
            colours the field through a discrete `BoundaryNorm` whose class
            edges match `styles.classify` on the same data.
        """
        arr = np.arange(100.0).reshape(10, 10)
        glyph = ArrayGlyph(arr)
        glyph.plot(classify=Classify(scheme="quantiles", k=4))
        assert isinstance(glyph.im.norm, mcolors.BoundaryNorm), (
            "ArrayGlyph should classify through a BoundaryNorm"
        )
        edges, _ = classify(arr, "quantiles", k=4)
        assert np.allclose(glyph.im.norm.boundaries, edges), (
            "raster class edges should match styles.classify"
        )

    def test_array_glyph_rejects_loose_scheme_kwarg(self):
        """A loose `scheme=` on the `ArrayGlyph` constructor still raises (#351).

        Test scenario:
            Even though `ArrayGlyph` now classifies via `classify=Classify(...)`,
            a loose `scheme=` keyword is still rejected with the pointer to the
            grouped parameter object (`scheme`/`k` remain in the grouped-kwarg
            hints), so the migration message is preserved.
        """
        arr = np.arange(9).reshape(3, 3).astype(float)
        with pytest.raises(ValueError, match="moved onto a grouped parameter object"):
            ArrayGlyph(arr, scheme="quantiles")

    def test_mesh_glyph_rejects_scheme(self):
        """`MeshGlyph` rejects `scheme` instead of silently ignoring it.

        Test scenario:
            MeshGlyph bypasses `_prepare_scalar_mapping`, so `scheme` is
            rejected at construction.
        """
        with pytest.raises(ValueError, match="moved onto a grouped parameter object"):
            MeshGlyph(
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([[0, 1, 2]]),
                scheme="quantiles",
            )

    def test_kde_glyph_rejects_scheme(self):
        """`KDEGlyph` rejects `scheme` (its `levels` owns discretisation).

        Test scenario:
            KDEGlyph is excluded from the classification options, so `scheme`
            is rejected at construction.
        """
        rng = np.random.default_rng(0)
        kx = rng.normal(size=20)
        ky = rng.normal(size=20)
        with pytest.raises(ValueError, match="moved onto a grouped parameter object"):
            KDEGlyph(kx, ky, scheme="quantiles")


class TestSchemeConflictWarnings:
    """Tests for the L2 conflict warnings when `scheme` overrides options."""

    def test_warns_on_conflicting_color_scale(self):
        """Setting `scheme` with a non-linear `color_scale` warns.

        Test scenario:
            `scheme` owns the norm, so `color_scale="midpoint"` is ignored —
            and a warning says so.
        """
        glyph = ScatterGlyph(np.arange(5.0), np.zeros(5), values=np.arange(5.0))
        with pytest.warns(UserWarning, match="color_scale"):
            glyph.plot(
                classify=Classify(scheme="quantiles"), color=ColorScaling.midpoint()
            )

    def test_warns_on_conflicting_levels(self):
        """Setting `scheme` together with `levels` warns.

        Test scenario:
            The classification scheme determines the bins, so `levels` is
            ignored — and a warning says so.
        """
        glyph = ScatterGlyph(np.arange(5.0), np.zeros(5), values=np.arange(5.0))
        with pytest.warns(UserWarning, match="levels"):
            glyph.plot(classify=Classify(scheme="quantiles"), contour=Contour(levels=4))

    def test_no_warning_without_conflict(self):
        """A plain `scheme` (default color_scale, no levels) does not warn.

        Test scenario:
            `scheme="quantiles"` alone produces no conflict warning.
        """
        import warnings as _warnings

        glyph = ScatterGlyph(np.arange(5.0), np.zeros(5), values=np.arange(5.0))
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            glyph.plot(classify=Classify(scheme="quantiles"))


class TestClassifyTwoDimensional:
    """Tests for `classify` on a 2-D values array (raster-shaped input)."""

    def test_2d_values_flattened_for_edges(self):
        """A 2-D values array is classified on its flattened finite values.

        Test scenario:
            A 10x10 grid of 0..99 yields the same quantile edges as the
            equivalent 1-D ramp.
        """
        grid = np.arange(100.0).reshape(10, 10)
        edges_2d, _ = classify(grid, "quantiles", k=4)
        edges_1d, _ = classify(np.arange(100.0), "quantiles", k=4)
        assert np.allclose(edges_2d, edges_1d), (
            f"2-D edges {edges_2d} should match 1-D edges {edges_1d}"
        )

    def test_2d_values_with_non_finite(self):
        """Non-finite cells in a 2-D array are ignored when binning.

        Test scenario:
            A grid with a NaN cell produces the same edges as the clean grid.
        """
        clean = np.arange(1.0, 101.0).reshape(10, 10)
        dirty = clean.copy()
        dirty[0, 0] = np.nan
        edges_clean, _ = classify(clean, "equal_interval", k=5)
        edges_dirty, _ = classify(dirty, "equal_interval", k=5)
        # equal_interval depends on min/max; the NaN cell was the min (1.0),
        # so removing it shifts the lower edge — assert the upper edge holds
        # and edges remain strictly increasing.
        assert np.isclose(edges_clean[-1], edges_dirty[-1]), "Upper edge should match"
        assert np.all(np.diff(edges_dirty) > 0), "Edges must stay strictly increasing"


class TestVectorGlyphScheme:
    """Integration tests for `scheme` through VectorGlyph."""

    @pytest.fixture()
    def field(self):
        """A small vector field with a spread of magnitudes.

        Returns:
            tuple[np.ndarray, ...]: x, y, u, v arrays on a 4x4 grid.
        """
        x, y = np.meshgrid(np.arange(4.0), np.arange(4.0))
        rng = np.random.default_rng(3)
        u = rng.uniform(0.1, 5.0, size=x.shape)
        v = rng.uniform(0.1, 5.0, size=x.shape)
        return x, y, u, v

    def test_quiver_scheme_uses_boundary_norm(self, field):
        """A classified quiver colours via a discrete BoundaryNorm.

        Test scenario:
            `scheme="quantiles"` makes the Quiver mappable use a BoundaryNorm
            with k+1 boundaries while still carrying the raw magnitude array.
        """
        x, y, u, v = field
        glyph = VectorGlyph(x, y, u, v)
        _, _, im = glyph.plot(classify=Classify(scheme="quantiles", k=5), kind="quiver")
        assert isinstance(im.norm, mcolors.BoundaryNorm), (
            "scheme should set a BoundaryNorm"
        )
        assert len(im.norm.boundaries) == 6, "k=5 should give 6 boundaries"
        assert np.allclose(im.get_array(), np.hypot(u, v).ravel()), (
            "Quiver should carry the raw magnitude array"
        )

    def test_barbs_scheme_discrete_colorbar(self, field):
        """A classified barbs plot draws a discrete colorbar.

        Test scenario:
            The colorbar norm is the discrete BoundaryNorm built from the
            magnitude classification.
        """
        x, y, u, v = field
        glyph = VectorGlyph(x, y, u, v)
        glyph.plot(classify=Classify(scheme="equal_interval", k=4), kind="barbs")
        assert glyph.cbar is not None, "A colorbar should be drawn"
        assert isinstance(glyph.cbar.norm, mcolors.BoundaryNorm), (
            "Colorbar norm should be discrete"
        )

    def test_streamplot_scheme_uses_boundary_norm(self):
        """A classified streamplot colours its lines via a BoundaryNorm.

        Test scenario:
            On a regular grid with a magnitude gradient, `scheme="quantiles"`
            makes the streamplot's `LineCollection` use a discrete
            BoundaryNorm and draws a discrete colorbar.
        """
        y, x = np.mgrid[0:6, 0:6].astype(float)
        u = x + 1.0
        v = y + 1.0
        glyph = VectorGlyph(x, y, u, v)
        _, _, im = glyph.plot(
            classify=Classify(scheme="quantiles", k=4), kind="streamplot"
        )
        assert isinstance(im.norm, mcolors.BoundaryNorm), (
            "scheme should set a BoundaryNorm"
        )
        assert len(im.norm.boundaries) == 5, "k=4 should give 5 boundaries"
        assert glyph.cbar is not None, "A discrete colorbar should be drawn"
        assert isinstance(glyph.cbar.norm, mcolors.BoundaryNorm), (
            "Colorbar norm should be discrete"
        )

    def test_scheme_none_regression(self, field):
        """`scheme=None` keeps the continuous vector colouring.

        Test scenario:
            Without a scheme the quiver is not BoundaryNorm-normalised.
        """
        x, y, u, v = field
        glyph = VectorGlyph(x, y, u, v)
        _, _, im = glyph.plot(kind="quiver")
        assert not isinstance(im.norm, mcolors.BoundaryNorm), (
            "No scheme -> no BoundaryNorm"
        )
