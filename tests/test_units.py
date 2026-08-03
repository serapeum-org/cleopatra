"""Tests for the units layer: ``convert_units`` and preset unit labels."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cleopatra.styling.colors import DATA_STYLES, convert_units


class TestConvertUnits:
    """Tests for ``cleopatra.styling.colors.convert_units``."""

    @pytest.mark.parametrize(
        "value, src, dst, expected",
        [
            (273.15, "K", "celsius", 0.0),
            (0.0, "celsius", "K", 273.15),
            (100.0, "celsius", "fahrenheit", 212.0),
            (32.0, "fahrenheit", "celsius", 0.0),
            (273.15, "K", "fahrenheit", 32.0),
            (32.0, "fahrenheit", "K", 273.15),
        ],
        ids=["K->C", "C->K", "C->F", "F->C", "K->F", "F->K"],
    )
    def test_known_conversions(self, value, src, dst, expected):
        """Each known temperature pair converts to the expected value.

        Args:
            value: Input magnitude.
            src: Source unit alias.
            dst: Target unit alias.
            expected: Expected converted value.

        Test scenario:
            The affine table maps each supported unit pair correctly.
        """
        result = float(convert_units(np.array([value]), src, dst)[0])
        assert result == pytest.approx(expected), f"{value}{src}->{dst}: expected {expected}, got {result}"

    @pytest.mark.parametrize("alias", ["K", "kelvin", "Kelvin", " KELVIN "])
    def test_aliases_are_normalised(self, alias):
        """Unit aliases (case/whitespace) resolve to the same conversion.

        Args:
            alias: A spelling of Kelvin.

        Test scenario:
            All Kelvin spellings convert 273.15 K to 0 C.
        """
        result = float(convert_units(np.array([273.15]), alias, "celsius")[0])
        assert result == pytest.approx(0.0), f"alias {alias!r} should convert like 'kelvin', got {result}"

    def test_none_units_is_noop(self):
        """A missing unit returns the data unchanged.

        Test scenario:
            ``from_units=None`` skips conversion entirely.
        """
        data = np.array([1.0, 2.0])
        assert convert_units(data, None, "celsius").tolist() == [1.0, 2.0], "None source unit must be a no-op"

    def test_same_units_is_noop(self):
        """Identical source and target units skip conversion.

        Test scenario:
            Converting celsius to celsius returns the input unchanged.
        """
        data = np.array([5.0, 10.0])
        assert convert_units(data, "celsius", "degC").tolist() == [5.0, 10.0], "same-unit conversion must be a no-op"

    def test_unknown_pair_warns_and_passes_through(self):
        """An unsupported unit pair warns and leaves the data unchanged.

        Test scenario:
            A non-temperature unit yields a ``UserWarning`` and the original
            values (styling never crashes on an unknown unit).
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = convert_units(np.array([1.0]), "meters", "celsius")
        assert len(caught) == 1, f"expected exactly one warning, got {len(caught)}"
        assert result.tolist() == [1.0], f"unknown pair should pass through unchanged, got {result.tolist()}"


class TestPresetUnitLabels:
    """The vendored weather presets bake a declared unit into their label."""

    def test_temperature_2m_label_carries_unit(self):
        """A unit-carrying preset shows its unit in brackets in the label.

        Test scenario:
            ``temperature_2m`` declares ``celsius`` upstream, so its layer label
            ends with ``[celsius]`` and the unit is kept on the layer.
        """
        layer = DATA_STYLES["temperature_2m"]["temperature_2m"]
        assert layer["label"].endswith("[celsius]"), f"label should carry the unit: {layer['label']!r}"
        assert layer.get("units") == "celsius", f"the unit should be kept on the layer, got {layer.get('units')!r}"

    def test_unitless_preset_label_has_no_brackets(self):
        """A preset without a declared unit keeps a plain label.

        Test scenario:
            ``elevation`` carries no unit, so its label has no ``[...]`` suffix
            and no ``units`` key.
        """
        layer = DATA_STYLES["elevation"]["elevation"]
        assert "[" not in layer["label"], f"unitless preset label should have no unit suffix: {layer['label']!r}"
        assert "units" not in layer, "a unitless preset should not carry a units key"
