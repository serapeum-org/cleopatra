"""Tests for the `ColorScaling` grouped colour-scale object."""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
import pytest

from cleopatra.styling.params import CellValues, Classify, Contour, DataStyle
from cleopatra.styling.scaling import ColorScaling


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
        ],
    )
    def test_variant_emits_all_six_keys(self, scale, key):
        """Every variant emits the full six-key option dict (full-scale reset).

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
        }, f"expected all six keys, got {set(options)}"
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
