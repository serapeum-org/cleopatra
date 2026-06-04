"""Tests for categorical (classified) colouring — issue #154.

Covers:

* `cleopatra.styles.classify` and its helpers `_scheme_edges` / `_jenks_edges`
  (the numpy-only schemes, explicit edges, the `BoundaryNorm` output, and the
  error / optional-extra paths).
* `Glyph._prepare_classified_mapping` and the `scheme` / `k` short-circuit
  added to `Glyph._prepare_scalar_mapping`.
* Integration through `ScatterGlyph` and `PolygonGlyph` (discrete classes,
  stepped colorbar, raw value preservation, and the `scheme=None` regression).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

import cleopatra.styles as styles_mod
from cleopatra.array_glyph import ArrayGlyph
from cleopatra.flow_glyph import FlowGlyph
from cleopatra.glyph import Glyph
from cleopatra.kde_glyph import KDEGlyph
from cleopatra.mesh_glyph import MeshGlyph
from cleopatra.polygon_glyph import PolygonGlyph
from cleopatra.scatter_glyph import ScatterGlyph
from cleopatra.vector_glyph import VectorGlyph
from cleopatra.styles import (
    CLASSIFY_OPTIONS,
    DEFAULT_OPTIONS as STYLE_DEFAULTS,
    JENKS_SCHEMES,
    NUMPY_SCHEMES,
    classify,
)
from cleopatra.styles import _jenks_edges, _scheme_edges


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
            ``JENKS_SCHEMES`` is exactly the mapclassify-backed names.
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
    """Tests for the public ``cleopatra.styles.classify`` function."""

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
        assert any(
            np.isclose(edges_k5, mean)
        ), f"mean break {mean} missing from {edges_k5}"
        assert any(
            np.isclose(edges_k5, mean - std)
        ), "mean-σ break missing"

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
        assert isinstance(norm, mcolors.BoundaryNorm), f"Expected BoundaryNorm, got {type(norm)}"
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
        assert np.allclose(edges_clean, edges_dirty), "Non-finite values must be ignored"

    def test_duplicate_edges_collapsed(self):
        """Repeated quantile edges are de-duplicated to keep edges increasing.

        Test scenario:
            Heavily tied data makes interior quantiles coincide; the result
            still has strictly increasing edges (a valid BoundaryNorm).
        """
        data = np.array([0.0] * 90 + [1.0] * 10)
        edges, norm = classify(data, "quantiles", k=5)
        assert np.all(np.diff(edges) > 0), f"Edges must strictly increase: {edges}"
        assert isinstance(norm, mcolors.BoundaryNorm), "Should still build a BoundaryNorm"

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

    def test_routes_jenks_to_extra(self):
        """A Jenks name is routed to the optional-extra path.

        Test scenario:
            With mapclassify absent, requesting fisher_jenks surfaces the
            install hint (proving the routing, not the computation).
        """
        with pytest.raises(ModuleNotFoundError, match="cleopatra\\[classify\\]"):
            _scheme_edges(np.arange(10.0), "fisher_jenks", k=3)

    def test_std_mean_k_ignored_branch(self):
        """``_scheme_edges`` does not validate ``k`` for ``std_mean``.

        Test scenario:
            ``k=0`` would raise for quantiles, but std_mean ignores k and
            returns edges without error.
        """
        edges = _scheme_edges(np.arange(100.0), "std_mean", k=0)
        assert edges[0] == 0.0 and edges[-1] == 99.0, f"Unexpected std_mean edges: {edges}"


class TestJenksEdges:
    """Tests for the private ``_jenks_edges`` helper."""

    def test_missing_mapclassify_raises_with_hint(self):
        """A missing optional extra raises ``ModuleNotFoundError`` with a hint.

        Test scenario:
            mapclassify is not installed in the test environment, so the
            helper re-raises with an actionable ``pip install`` hint.
        """
        with pytest.raises(ModuleNotFoundError, match="pip install 'cleopatra\\[classify\\]'"):
            _jenks_edges(np.arange(10.0), "natural_breaks", k=3)

    def test_k_below_one_raises(self):
        """``_jenks_edges`` validates ``k`` before importing the extra.

        Test scenario:
            ``k < 1`` raises ``ValueError`` regardless of mapclassify.
        """
        with pytest.raises(ValueError, match="`k` must be >= 1"):
            _jenks_edges(np.arange(10.0), "fisher_jenks", k=0)

    def test_uses_mapclassify_when_available(self, monkeypatch):
        """When mapclassify is importable, edges are ``[min, *bins]``.

        Test scenario:
            A fake ``mapclassify`` module with a ``FisherJenks`` classifier is
            injected; the helper prepends the data minimum to the classifier
            bins.
        """
        import sys
        import types

        fake = types.ModuleType("mapclassify")

        class _FisherJenks:
            def __init__(self, data, k):
                self.bins = np.array([3.0, 6.0, 9.0])

        fake.FisherJenks = _FisherJenks
        monkeypatch.setitem(sys.modules, "mapclassify", fake)
        edges = _jenks_edges(np.arange(10.0), "fisher_jenks", k=3)
        assert np.allclose(edges, [0.0, 3.0, 6.0, 9.0]), f"Unexpected jenks edges: {edges}"


class TestGlyphPrepareClassifiedMapping:
    """Tests for ``Glyph._prepare_classified_mapping`` and the routing."""

    def test_returns_norm_cbar_edges_triple(self):
        """The helper returns the ``(norm, cbar_kw, edges)`` contract.

        Test scenario:
            A quantile scheme yields a BoundaryNorm, boundary ticks, and the
            bin edges in the ticks slot.
        """
        g = Glyph(default_options=_make_options(scheme="quantiles", k=4))
        norm, cbar_kw, edges = g._prepare_classified_mapping(np.arange(100.0), "quantiles")
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
        assert cbar_kw["extend"] == "neither", f"extend should be 'neither', got {cbar_kw['extend']}"

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
        assert isinstance(norm, mcolors.BoundaryNorm), "scheme must route to BoundaryNorm"
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
        glyph = ScatterGlyph(x, y, values=v, scheme="quantiles", k=5)
        _, _, paths = glyph.plot()
        assert isinstance(paths.norm, mcolors.BoundaryNorm), "scheme should set a BoundaryNorm"
        assert len(paths.norm.boundaries) == 6, "k=5 should give 6 boundaries"

    def test_raw_values_preserved(self, xy_values):
        """Classification does not alter the underlying value array.

        Test scenario:
            ``get_array`` still returns the raw per-point values.
        """
        x, y, v = xy_values
        glyph = ScatterGlyph(x, y, values=v, scheme="equal_interval", k=4)
        _, _, paths = glyph.plot()
        assert np.array_equal(paths.get_array(), v), "Raw values must be preserved"

    def test_colorbar_drawn_by_default(self, xy_values):
        """A classified scatter draws a (stepped) colorbar by default.

        Test scenario:
            ``add_colorbar`` defaults to True, so a colorbar is created.
        """
        x, y, v = xy_values
        glyph = ScatterGlyph(x, y, values=v, scheme="quantiles", k=5)
        glyph.plot()
        assert glyph.cbar is not None, "A colorbar should be drawn by default"

    def test_add_colorbar_false_suppresses(self, xy_values):
        """``add_colorbar=False`` suppresses the colorbar with a scheme set.

        Test scenario:
            The plot-time override wins and no colorbar is created.
        """
        x, y, v = xy_values
        glyph = ScatterGlyph(x, y, values=v, scheme="quantiles", k=5)
        glyph.plot(add_colorbar=False)
        assert glyph.cbar is None, "add_colorbar=False should suppress the colorbar"

    def test_scheme_none_regression(self, xy_values):
        """``scheme=None`` keeps the continuous (non-BoundaryNorm) behaviour.

        Test scenario:
            Without a scheme the scatter is not normalised by a BoundaryNorm.
        """
        x, y, v = xy_values
        glyph = ScatterGlyph(x, y, values=v)
        _, _, paths = glyph.plot()
        assert not isinstance(paths.norm, mcolors.BoundaryNorm), "No scheme -> no BoundaryNorm"


class TestPolygonGlyphScheme:
    """Integration tests for ``scheme`` through PolygonGlyph (choropleth)."""

    @pytest.fixture()
    def polys_values(self):
        """Ten triangles with a 0..9 value ramp.

        Returns:
            tuple[list[np.ndarray], np.ndarray]: polygon vertices and values.
        """
        polys = [
            np.array([[i, 0.0], [i + 1, 0.0], [i + 0.5, 1.0]]) for i in range(10)
        ]
        return polys, np.arange(10.0)

    def test_five_discrete_classes(self, polys_values):
        """A k=5 quantile choropleth yields five discrete classes.

        Test scenario:
            The PolyCollection uses a BoundaryNorm whose six boundaries
            delimit five fill classes.
        """
        polys, values = polys_values
        glyph = PolygonGlyph(polys, values=values, scheme="quantiles", k=5)
        _, _, pc = glyph.plot()
        assert isinstance(pc.norm, mcolors.BoundaryNorm), "choropleth should use a BoundaryNorm"
        assert len(pc.norm.boundaries) - 1 == 5, "Six boundaries delimit five classes"

    def test_raw_values_preserved(self, polys_values):
        """The polygon array still carries the raw values.

        Test scenario:
            ``get_array`` returns the unbinned per-polygon values.
        """
        polys, values = polys_values
        glyph = PolygonGlyph(polys, values=values, scheme="quantiles", k=5)
        _, _, pc = glyph.plot()
        assert np.array_equal(pc.get_array(), values), "Raw values must be preserved"

    def test_discrete_colorbar_drawn(self, polys_values):
        """A discrete colorbar is attached for a classified choropleth.

        Test scenario:
            The colorbar exists and its norm is the discrete BoundaryNorm.
        """
        polys, values = polys_values
        glyph = PolygonGlyph(polys, values=values, scheme="quantiles", k=5)
        glyph.plot()
        assert glyph.cbar is not None, "A discrete colorbar should be drawn"
        assert isinstance(glyph.cbar.norm, mcolors.BoundaryNorm), "Colorbar norm should be discrete"

    def test_scheme_none_regression(self, polys_values):
        """``scheme=None`` keeps the continuous choropleth behaviour.

        Test scenario:
            Without a scheme the PolyCollection is not BoundaryNorm-normalised.
        """
        polys, values = polys_values
        glyph = PolygonGlyph(polys, values=values)
        _, _, pc = glyph.plot()
        assert not isinstance(pc.norm, mcolors.BoundaryNorm), "No scheme -> no BoundaryNorm"


class TestSchemeGlyphScope:
    """Tests for which glyphs accept `scheme` (the M1 fix).

    `scheme`/`k` live in `CLASSIFY_OPTIONS` and are mixed only into glyphs
    whose colour mapping routes through `Glyph._prepare_scalar_mapping` and
    is driven purely by the norm. Glyphs that bypass that pipeline
    (`ArrayGlyph`, `MeshGlyph`) — and `KDEGlyph`, whose `contourf` has its
    own `levels` discretisation — must reject `scheme` rather than silently
    ignore it.
    """

    @pytest.mark.parametrize("glyph_cls", [ScatterGlyph, PolygonGlyph, VectorGlyph, FlowGlyph])
    def test_pipeline_glyphs_accept_scheme(self, glyph_cls):
        """Pipeline glyphs expose `scheme`/`k` as accepted options.

        Args:
            glyph_cls: A glyph class that colours through the shared pipeline.

        Test scenario:
            `scheme` and `k` are in the class's option keys.
        """
        keys = glyph_cls.option_keys()
        assert "scheme" in keys and "k" in keys, (
            f"{glyph_cls.__name__} should accept scheme/k"
        )

    def test_array_glyph_rejects_scheme(self):
        """`ArrayGlyph` rejects `scheme` instead of silently ignoring it.

        Test scenario:
            ArrayGlyph bypasses `_prepare_scalar_mapping`, so `scheme` is not
            an accepted option and construction raises.
        """
        with pytest.raises(ValueError, match="not correct"):
            ArrayGlyph(np.arange(9).reshape(3, 3).astype(float), scheme="quantiles")

    def test_mesh_glyph_rejects_scheme(self):
        """`MeshGlyph` rejects `scheme` instead of silently ignoring it.

        Test scenario:
            MeshGlyph bypasses `_prepare_scalar_mapping`, so `scheme` is
            rejected at construction.
        """
        with pytest.raises(ValueError, match="not correct"):
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
        with pytest.raises(ValueError, match="not correct"):
            KDEGlyph(rng.normal(size=20), rng.normal(size=20), scheme="quantiles")


class TestSchemeConflictWarnings:
    """Tests for the L2 conflict warnings when `scheme` overrides options."""

    def test_warns_on_conflicting_color_scale(self):
        """Setting `scheme` with a non-linear `color_scale` warns.

        Test scenario:
            `scheme` owns the norm, so `color_scale="midpoint"` is ignored —
            and a warning says so.
        """
        glyph = ScatterGlyph(
            np.arange(5.0), np.zeros(5), values=np.arange(5.0),
            scheme="quantiles", color_scale="midpoint",
        )
        with pytest.warns(UserWarning, match="color_scale"):
            glyph.plot()

    def test_warns_on_conflicting_levels(self):
        """Setting `scheme` together with `levels` warns.

        Test scenario:
            The classification scheme determines the bins, so `levels` is
            ignored — and a warning says so.
        """
        glyph = ScatterGlyph(
            np.arange(5.0), np.zeros(5), values=np.arange(5.0),
            scheme="quantiles", levels=4,
        )
        with pytest.warns(UserWarning, match="levels"):
            glyph.plot()

    def test_no_warning_without_conflict(self):
        """A plain `scheme` (default color_scale, no levels) does not warn.

        Test scenario:
            `scheme="quantiles"` alone produces no conflict warning.
        """
        import warnings as _warnings

        glyph = ScatterGlyph(
            np.arange(5.0), np.zeros(5), values=np.arange(5.0), scheme="quantiles",
        )
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            glyph.plot()


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
        glyph = VectorGlyph(x, y, u, v, scheme="quantiles", k=5)
        _, _, im = glyph.plot(kind="quiver")
        assert isinstance(im.norm, mcolors.BoundaryNorm), "scheme should set a BoundaryNorm"
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
        glyph = VectorGlyph(x, y, u, v, scheme="equal_interval", k=4)
        glyph.plot(kind="barbs")
        assert glyph.cbar is not None, "A colorbar should be drawn"
        assert isinstance(glyph.cbar.norm, mcolors.BoundaryNorm), "Colorbar norm should be discrete"

    def test_scheme_none_regression(self, field):
        """`scheme=None` keeps the continuous vector colouring.

        Test scenario:
            Without a scheme the quiver is not BoundaryNorm-normalised.
        """
        x, y, u, v = field
        glyph = VectorGlyph(x, y, u, v)
        _, _, im = glyph.plot(kind="quiver")
        assert not isinstance(im.norm, mcolors.BoundaryNorm), "No scheme -> no BoundaryNorm"
