"""Tests for the `ColorScaling` grouped colour-scale object."""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
import pytest

from cleopatra.styling.params import CellValues, Classify, Contour, DataStyle
from cleopatra.styling.scaling import (
    ColorScaling,
    _auto_linthresh,
    _log_tick_positions,
    _symlog_tick_positions,
)
from cleopatra.styling.styles import ColorScale


class TestColorScalingToOptions:
    """`ColorScaling.to_options` emits the flat colour-scale keys."""

    def test_non_midpoint_variant_does_not_leak_a_method_into_midpoint(self):
        """A non-midpoint scale emits the numeric `midpoint` default, not a method.

        Test scenario:
            Regression: the `midpoint` field once shadowed the `midpoint()`
            variant constructor, so `power()`/`linear()` emitted a bound
            method as the `midpoint` option instead of `0`.
        """
        options = ColorScaling.power(gamma=0.7).to_options()
        assert options["midpoint"] == 0, (
            f"midpoint should default to 0, got {options['midpoint']!r}"
        )
        assert isinstance(options["midpoint"], (int, float)), (
            f"midpoint must be numeric, got {type(options['midpoint'])}"
        )

    def test_midpoint_variant_carries_its_centre(self):
        """`ColorScaling.midpoint(at=X)` emits `X` as the `midpoint` option."""
        assert ColorScaling.midpoint(at=42).to_options()["midpoint"] == 42

    @pytest.mark.parametrize(
        "scale, key",
        [
            (ColorScaling.power(gamma=0.3), "color_scale"),
            (ColorScaling.boundary(bounds=[0, 1, 2]), "bounds"),
            (ColorScaling.sym_log(threshold=0.01, scale=0.1), "line_threshold"),
            (ColorScaling.log(), "color_scale"),
            (ColorScaling.equalize(samples=256), "samples"),
        ],
    )
    def test_variant_emits_all_seven_keys(self, scale, key):
        """Every variant emits the full seven-key option dict (full-scale reset).

        Args:
            scale: A `ColorScaling` variant.
            key: A key expected in the emitted options.
        """
        options = scale.to_options()
        assert set(options) == {
            "color_scale",
            "gamma",
            "line_threshold",
            "line_scale",
            "bounds",
            "midpoint",
            "samples",
        }, f"expected all seven keys, got {set(options)}"
        assert key in options


class TestColorScalingBuildNorm:
    """`ColorScaling.build_norm` reproduces the scale's matplotlib norm."""

    def test_linear_without_levels_has_no_norm(self):
        """A plain linear scale returns no norm and passes ticks through."""
        norm, cbar_kw = ColorScaling.linear().build_norm(np.array([0.0, 5.0, 10.0]))
        assert norm is None, "linear scale should have no explicit norm"
        assert cbar_kw["extend"] == "neither"

    def test_midpoint_builds_a_midpoint_norm(self):
        """The midpoint scale builds a `MidpointNormalize` centred at `at`."""
        norm, _ = ColorScaling.midpoint(at=2.0).build_norm(np.array([0.0, 4.0]))
        assert type(norm).__name__ == "MidpointNormalize"
        assert norm.midpoint == 2.0, f"midpoint should be 2.0, got {norm.midpoint}"

    def test_power_builds_a_power_norm(self):
        """The power scale builds a `PowerNorm` with the given gamma."""
        norm, _ = ColorScaling.power(gamma=0.5).build_norm(np.array([0.0, 10.0]))
        assert isinstance(norm, mcolors.PowerNorm)
        assert norm.gamma == 0.5

    def test_log_builds_a_log_norm(self):
        """The log scale builds a `LogNorm` over the positive tick range."""
        norm, cbar_kw = ColorScaling.log().build_norm(np.array([1.0, 10.0, 100.0]))
        assert isinstance(norm, mcolors.LogNorm)
        assert (norm.vmin, norm.vmax) == (1.0, 100.0), (
            f"LogNorm should span the ticks, got ({norm.vmin}, {norm.vmax})"
        )
        assert cbar_kw["extend"] == "neither"

    def test_log_on_non_positive_range_raises(self):
        """A log scale whose range starts at zero raises, steering at sym_log."""
        scale = ColorScaling.log()
        ticks = np.array([0.0, 10.0, 100.0])
        with pytest.raises(ValueError, match="strictly-positive"):
            scale.build_norm(ticks)

    def test_log_on_constant_positive_data_widens_the_range(self):
        """A constant positive field (single tick) builds a LogNorm, not a crash.

        Test scenario:
            Uniform data yields one tick, so vmin == vmax. A log scale cannot
            span a zero-width range; the branch widens it (like the data-style
            path) rather than raising, matching the other scale kinds.
        """
        norm, _ = ColorScaling.log().build_norm(np.array([5.0]))
        assert isinstance(norm, mcolors.LogNorm)
        assert norm.vmin == 5.0, f"vmin should stay 5.0, got {norm.vmin}"
        assert norm.vmax == 6.0, f"vmax should widen to 6.0, got {norm.vmax}"

    def test_log_on_constant_negative_data_reports_real_bounds(self):
        """A constant non-positive field raises with its real bound, not a widened one.

        Test scenario:
            The degenerate-range widening applies only to strictly-positive
            constants, so an all-negative field is not widened before the error
            is built -- the message reports the real value and steers at sym_log.
        """
        scale = ColorScaling.log()
        ticks = np.array([-5.0])
        with pytest.raises(ValueError, match=r"vmin=-5\.0, vmax=-5\.0"):
            scale.build_norm(ticks)

    def test_log_options_round_trip(self):
        """`log()` emits `color_scale='lognorm'` and reconstructs to LOGNORM."""
        opts = ColorScaling.log().to_options()
        assert opts["color_scale"] == "lognorm"
        assert ColorScaling.from_options(opts).kind.name == "LOGNORM"

    def test_sym_log_bar_ticks_are_scale_aware_and_signed(self):
        """sym_log places decade-aligned bar ticks and keeps negative signs (#335).

        Test scenario:
            The linear ladder ([-24, 744] here) supplies vmin/vmax, but the bar
            ticks are the symlog decades within that range, and the formatter
            labels a negative decade with its sign -- unlike LogFormatter, which
            blanked non-decades and dropped the sign of -10.
        """
        _, cbar_kw = ColorScaling.sym_log(threshold=10.0, scale=1.0).build_norm(
            np.array([-24.0, 0.0, 744.0])
        )
        ticks = np.asarray(cbar_kw["ticks"])
        assert ticks.size >= 2, f"expected several bar ticks, got {ticks.tolist()}"
        assert ticks.min() >= -24.0, f"tick below vmin: {ticks.tolist()}"
        assert ticks.max() <= 744.0, f"tick above vmax: {ticks.tolist()}"
        assert ticks.min() < 0.0, f"a below-zero range should span a negative tick: {ticks.tolist()}"
        nonzero = ticks[ticks != 0.0]
        decades = np.log10(np.abs(nonzero))
        assert np.allclose(decades, np.round(decades)), f"non-decade ticks: {ticks.tolist()}"
        fmt = cbar_kw["format"]
        assert fmt(-10.0) == "-10", f"negative decade must keep its sign, got {fmt(-10.0)!r}"
        assert fmt(100.0) == "100", f"expected '100', got {fmt(100.0)!r}"

    def test_sym_log_default_auto_derives_linthresh_from_the_range(self):
        """The default (no `threshold`) sizes the norm's `linthresh` to the data (#337).

        Test scenario:
            On a wide terrain range [-24, 744] the old fixed `0.0001` default
            pushed almost everything into the log region, so the bar filled with
            sub-scale near-zero decades. With no explicit `threshold`, the norm
            now derives `linthresh` from the range (1% of the peak magnitude),
            and the bar shows only decades near the data's magnitude -- this is a
            norm-level change (the rendered image), not just the tick labels. The
            `|tick| >= 1` assertion is specific to this large-magnitude range,
            not a general guarantee -- see
            `test_sym_log_auto_derivation_on_o1_range_still_shows_sub_scale_decades`.
        """
        norm, cbar_kw = ColorScaling.sym_log().build_norm(np.array([-24.0, 744.0]))
        assert isinstance(norm, mcolors.SymLogNorm)
        assert norm.linthresh == pytest.approx(_auto_linthresh(-24.0, 744.0)), (
            f"norm linthresh should be data-derived, got {norm.linthresh}"
        )
        assert norm.linthresh == pytest.approx(7.44), (
            f"1% of peak 744 should be 7.44, got {norm.linthresh}"
        )
        ticks = np.asarray(cbar_kw["ticks"])
        nonzero = ticks[ticks != 0.0]
        assert np.all(np.abs(nonzero) >= 1.0), (
            f"default bar should not show sub-scale near-zero decades, got {ticks.tolist()}"
        )

    def test_sym_log_explicit_threshold_overrides_auto_derivation(self):
        """An explicit `threshold` still wins, sub-scale decades and all (#337).

        Test scenario:
            The auto-derivation only applies when `threshold` is omitted. Passing
            the old `0.0001` explicitly must reproduce the old norm exactly --
            `linthresh` stays `0.0001` and the sub-unit decades reappear -- proving
            the caller's value is never overridden.
        """
        norm, cbar_kw = ColorScaling.sym_log(threshold=0.0001).build_norm(
            np.array([-24.0, 744.0])
        )
        assert norm.linthresh == 0.0001, (
            f"explicit threshold must not be auto-derived, got {norm.linthresh}"
        )
        ticks = np.asarray(cbar_kw["ticks"])
        assert np.any((np.abs(ticks) > 0.0) & (np.abs(ticks) < 1.0)), (
            f"an explicit tiny threshold should still expose sub-unit decades, got {ticks.tolist()}"
        )

    def test_auto_linthresh_tracks_the_peak_magnitude(self):
        """`_auto_linthresh` is 1% of the peak magnitude, with a positive floor."""
        assert _auto_linthresh(-24.0, 744.0) == pytest.approx(7.44)
        assert _auto_linthresh(-1000.0, 1000.0) == pytest.approx(10.0)
        assert _auto_linthresh(0.0, 0.0) == 1.0, "all-zero range needs a positive floor"

    def test_sym_log_auto_derivation_flows_through_from_options(self):
        """The flat `line_threshold=None` option derives `linthresh` too (#337).

        Test scenario:
            Glyphs reach the norm via `from_options` on their flat option dict,
            whose `line_threshold` now defaults to `None`. Rebuilding a
            sym-lognorm scale from a bare `{"color_scale": "sym-lognorm"}` (no
            `line_threshold`) must derive the threshold from the range, exactly
            like the `sym_log()` object path -- not fall back to a fixed value.
        """
        scaling = ColorScaling.from_options({"color_scale": "sym-lognorm"})
        assert scaling.line_threshold is None, (
            "flat default should defer to auto-derivation"
        )
        norm, _ = scaling.build_norm(np.array([-24.0, 744.0]))
        assert norm.linthresh == pytest.approx(_auto_linthresh(-24.0, 744.0)), (
            f"flat path should derive linthresh, got {norm.linthresh}"
        )

    def test_sym_log_auto_derivation_on_a_negative_only_range(self):
        """A negative-only range derives `linthresh` from `|vmin|` (#337).

        Test scenario:
            `_auto_linthresh` is the peak *magnitude*, so an all-negative range
            like [-744, -24] must size the band off `|vmin| = 744`, not the
            near-zero `vmax`.
        """
        norm, cbar_kw = ColorScaling.sym_log().build_norm(np.array([-744.0, -24.0]))
        assert norm.linthresh == pytest.approx(7.44), (
            f"|vmin|=744 should drive linthresh to 7.44, got {norm.linthresh}"
        )
        # Only one decade (-100) lands inside [-744, -24], so the bar ticks fall
        # back to the endpoints: the derivation sizes the norm correctly but a
        # narrow (<2-decade) negative-only span yields no decade ticks.
        assert np.asarray(cbar_kw["ticks"]).tolist() == [-744.0, -24.0], (
            f"expected the endpoint fallback, got {cbar_kw['ticks']}"
        )

    def test_sym_log_auto_derivation_on_o1_range_still_shows_sub_scale_decades(self):
        """An O(1) range straddling zero still shows sub-unit decades (#337).

        Test scenario:
            The derivation bounds how far the decades reach below the data; it
            does not pin the smallest decade to the data's scale. For [-5, 5] the
            1%-of-peak `linthresh` is 0.05, so the bar legitimately still shows
            sub-unit decades (0.1, 0.01) -- the honest counterpart to the
            wide-range case, and the limit of the auto-derivation.
        """
        norm, cbar_kw = ColorScaling.sym_log().build_norm(np.array([-5.0, 5.0]))
        assert norm.linthresh == pytest.approx(0.05), (
            f"1% of peak 5 should be 0.05, got {norm.linthresh}"
        )
        # The honest counterpart to the wide-range case: a sub-unit `linthresh`
        # keeps the linear band below the data's own scale, so the log region --
        # and any decade ticks in it -- reaches below 1 rather than stopping near
        # the data magnitude. Assert that invariant (robust) rather than the exact
        # locator tick set, which is matplotlib-version dependent.
        assert norm.linthresh < 1.0, (
            f"an O(1) straddle range keeps a sub-unit linear band, got {norm.linthresh}"
        )
        nonzero = np.asarray(cbar_kw["ticks"])
        nonzero = nonzero[nonzero != 0.0]
        decades = np.log10(np.abs(nonzero))
        assert np.allclose(decades, np.round(decades)), f"non-decade ticks: {nonzero.tolist()}"

    def test_sym_log_default_keeps_in_band_ticks_legible(self):
        """The auto default spaces the near-zero in-band ticks legibly (#337).

        Test scenario:
            Auto-deriving `linthresh` widens the linear band, so pairing it with
            the old tiny `linscale=0.001` crushed that band to a sliver and the
            in-band ticks (-1, 0, 1 on [-24, 744]) overprinted. With the auto
            `linscale` the surviving ticks map to distinct colour-bar positions;
            an explicit `scale` still reproduces the old (crushed) spacing.
        """
        norm, cbar_kw = ColorScaling.sym_log().build_norm(np.array([-24.0, 744.0]))
        positions = np.sort([float(norm(t)) for t in np.asarray(cbar_kw["ticks"])])
        assert np.diff(positions).min() > 0.01, (
            f"adjacent bar ticks must be legibly spaced, got positions {positions.tolist()}"
        )
        crushed_norm, crushed_kw = ColorScaling.sym_log(scale=0.001).build_norm(
            np.array([-24.0, 744.0])
        )
        crushed = np.sort([float(crushed_norm(t)) for t in np.asarray(crushed_kw["ticks"])])
        assert np.diff(crushed).min() < 0.001, (
            f"an explicit scale should still win (old crushed spacing), got {crushed.tolist()}"
        )

    def test_log_bar_ticks_are_decade_aligned(self):
        """log places decade bar ticks and a formatter that labels them (#335)."""
        _, cbar_kw = ColorScaling.log().build_norm(np.array([0.5, 744.0]))
        ticks = np.asarray(cbar_kw["ticks"])
        assert ticks.size >= 2, f"expected several bar ticks, got {ticks.tolist()}"
        assert ticks.min() >= 0.5, f"tick below vmin: {ticks.tolist()}"
        assert ticks.max() <= 744.0, f"tick above vmax: {ticks.tolist()}"
        decades = np.log10(ticks)
        assert np.allclose(decades, np.round(decades)), f"non-decade ticks: {ticks.tolist()}"
        assert cbar_kw["format"](10.0) == "10", "log formatter should label a decade"

    def test_non_linear_formatter_labels_arbitrary_positions(self):
        """The sym_log formatter labels any value, so set_ticks needs no set_ticklabels.

        Test scenario:
            The formatter is position-agnostic: a caller's non-decade tick (e.g.
            -5 or 42.5) is labelled with its plain value, which is what makes a
            later cbar.set_ticks([...]) readable without a paired set_ticklabels.
        """
        _, cbar_kw = ColorScaling.sym_log(threshold=10.0).build_norm(
            np.array([-24.0, 744.0])
        )
        fmt = cbar_kw["format"]
        assert fmt(-5.0) == "-5", f"expected '-5', got {fmt(-5.0)!r}"
        assert fmt(42.5) == "42.5", f"expected '42.5', got {fmt(42.5)!r}"

    def test_tick_positions_fall_back_to_the_ladder_when_sparse(self):
        """When no decade lands in range, the helpers return the caller's ladder."""
        ladder = np.array([1.0, 2.0, 3.0])
        assert _log_tick_positions(2.0, 5.0, ladder).tolist() == ladder.tolist(), (
            "log within one decade should fall back to the ladder"
        )
        assert _symlog_tick_positions(200.0, 500.0, 10.0, ladder).tolist() == (
            ladder.tolist()
        ), "symlog within one decade should fall back to the ladder"

    def test_formatter_normalizes_signed_zero(self):
        """The tick formatter renders a signed zero as '0', not '-0'."""
        fmt = ColorScaling.sym_log(threshold=10.0).build_norm(
            np.array([-24.0, 744.0])
        )[1]["format"]
        assert fmt(-0.0) == "0", f"signed zero should render '0', got {fmt(-0.0)!r}"
        assert fmt(0.0) == "0", f"zero should render '0', got {fmt(0.0)!r}"

    def test_log_ticks_stay_bounded_over_many_decades(self):
        """A log bar spanning many decades stays a handful of decade ticks, not hundreds."""
        _, cbar_kw = ColorScaling.log().build_norm(np.array([1e-6, 1e6]))
        ticks = np.asarray(cbar_kw["ticks"])
        # LogLocator strides decades on wide ranges (13 decades here -> ~7 ticks); 30
        # is generous headroom that only guards against an unbounded explosion. The
        # decade-alignment assertion below is the load-bearing check.
        assert 2 <= ticks.size <= 30, f"decade set should stay bounded, got {ticks.size}"
        decades = np.log10(ticks)
        assert np.allclose(decades, np.round(decades)), f"non-decade ticks: {ticks.tolist()}"


class TestColorScalingEqualize:
    """Tests for the continuous rank-equalising (`equalize`) colour scale."""

    @staticmethod
    def _skewed() -> np.ndarray:
        """A deep bulk plus a thin shallow tail, like bathymetry (seeded)."""
        rng = np.random.default_rng(0)
        return np.concatenate(
            [rng.normal(-3000, 400, 90_000), rng.normal(-300, 200, 10_000)]
        )

    def test_factory_carries_kind_and_samples(self):
        """`equalize()` sets the EQUALIZE kind and records its sample count."""
        scale = ColorScaling.equalize(samples=256)
        assert scale.kind is ColorScale.EQUALIZE, f"wrong kind: {scale.kind}"
        assert scale.samples == 256, f"wrong samples: {scale.samples}"

    def test_factory_rejects_too_few_samples(self):
        """`equalize(samples<2)` raises, since a CDF table needs two points."""
        with pytest.raises(ValueError, match="samples >= 2"):
            ColorScaling.equalize(samples=1)

    def test_every_decile_gets_an_even_share_of_the_ramp(self):
        """A skewed field's every decile receives ~10% of the colour ramp."""
        data = self._skewed()
        edges = np.percentile(data, np.arange(0, 101, 10))
        ticks = np.linspace(data.min(), data.max(), 8)
        norm, _ = ColorScaling.equalize().build_norm(ticks, values=data)
        shares = np.diff(norm(edges)) * 100
        assert np.allclose(shares, 10.0, atol=0.5), f"uneven ramp shares: {shares}"

    def test_colorbar_ticks_are_placed_at_quantiles(self):
        """The colour bar's ticks sit at the data's quantiles, not linearly."""
        data = self._skewed()
        ticks = np.linspace(data.min(), data.max(), 8)
        _, cbar_kw = ColorScaling.equalize().build_norm(ticks, values=data)
        expected = np.unique(np.quantile(data, np.linspace(0.0, 1.0, len(ticks))))
        assert np.allclose(cbar_kw["ticks"], expected), (
            f"ticks not at quantiles: {cbar_kw['ticks']}"
        )
        spacings = np.diff(cbar_kw["ticks"])
        assert spacings.std() > 0.0, "quantile ticks should not be evenly spaced"

    def test_constant_field_does_not_raise(self):
        """A constant field yields a degenerate linear norm instead of raising."""
        norm, _ = ColorScaling.equalize().build_norm(
            np.array([5.0]), values=np.full(100, 5.0)
        )
        assert isinstance(norm, mcolors.Normalize), f"unexpected norm: {norm!r}"

    def test_heavily_tied_field_does_not_raise(self):
        """A field that is almost all one value still builds a norm without error."""
        tied = np.array([0.0] * 95 + [1.0] * 5, dtype=float)
        norm, _ = ColorScaling.equalize().build_norm(np.array([0.0, 1.0]), values=tied)
        assert isinstance(norm, mcolors.Normalize), f"unexpected norm: {norm!r}"

    def test_missing_values_raise_a_clear_error(self):
        """Building the equalize norm without the data raises an actionable error."""
        with pytest.raises(ValueError, match="needs the data values"):
            ColorScaling.equalize().build_norm(np.array([0.0, 1.0]))

    def test_all_non_finite_values_raise(self):
        """A field with no finite values raises rather than ranking an empty set."""
        with pytest.raises(ValueError, match="no finite values"):
            ColorScaling.equalize().build_norm(
                np.array([0.0, 1.0]), values=np.array([np.nan, np.inf, -np.inf])
            )

    def test_samples_round_trips_through_options(self):
        """`samples` survives the flat-options round-trip."""
        restored = ColorScaling.from_options(
            ColorScaling.equalize(samples=256).to_options()
        )
        assert restored.samples == 256, f"samples lost: {restored.samples}"


class TestParamGroupsEmitOnlySetFields:
    """`Contour`/`CellValues`/`DataStyle`/`Classify` emit only the fields set."""

    def test_empty_groups_emit_nothing(self):
        """A group with no fields set emits an empty option dict."""
        assert Contour().to_options() == {}
        assert CellValues().to_options() == {}
        assert Classify().to_options() == {}

    def test_classify_emits_only_set_fields(self):
        """`Classify` emits scheme/k/category_legend_kwargs only when given."""
        assert Classify(scheme="quantiles").to_options() == {"scheme": "quantiles"}
        assert Classify(k=4).to_options() == {"k": 4}
        assert Classify(category_legend_kwargs={"loc": "upper left"}).to_options() == {
            "category_legend_kwargs": {"loc": "upper left"}
        }
        assert Classify(scheme="quantiles", k=4).to_options() == {
            "scheme": "quantiles",
            "k": 4,
        }

    def test_contour_and_cells_emit_only_set_fields(self):
        """`Contour`/`CellValues` emit only the fields explicitly provided."""
        assert Contour(levels=5).to_options() == {"levels": 5}
        assert Contour(labels=True, label_kw={"fmt": "%.2f"}).to_options() == {
            "labels": True,
            "label_kw": {"fmt": "%.2f"},
        }
        assert CellValues(show=True, size=8, background_threshold=0.5).to_options() == {
            "display_cell_value": True,
            "num_size": 8,
            "background_color_threshold": 0.5,
        }

    def test_datastyle_unset_omits_but_explicit_none_clears(self):
        """`DataStyle` omits unset fields but emits an explicit `None` (clear)."""
        assert DataStyle().to_options() == {}
        assert DataStyle(style=None).to_options() == {"style": None}
        assert DataStyle(style="dem", hillshade=True).to_options() == {
            "style": "dem",
            "hillshade": True,
        }
