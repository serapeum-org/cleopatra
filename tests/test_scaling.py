"""Tests for the `ColorScaling` grouped colour-scale object."""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
import pytest

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
