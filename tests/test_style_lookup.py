"""Tests for the GRIB shortName -> style lookup (``style_for_parameter``)."""

from __future__ import annotations

import pytest

from cleopatra.styling.colors import DATA_STYLES, _SHORTNAME_TO_STYLE, style_for_parameter


class TestStyleForParameter:
    """Tests for ``cleopatra.styling.colors.style_for_parameter``."""

    @pytest.mark.parametrize(
        "short_name, expected",
        [("2t", "temperature_2m"), ("aod550", "aerosol_optical_depth_550nm"), ("tp", "total_precipitation")],
    )
    def test_known_shortnames_resolve(self, short_name, expected):
        """A known GRIB shortName maps to its descriptive preset.

        Args:
            short_name: The GRIB shortName.
            expected: The descriptive preset name.

        Test scenario:
            The vendored map resolves canonical shortNames to live presets.
        """
        assert style_for_parameter(short_name) == expected, f"{short_name} should map to {expected}"

    def test_lookup_is_case_insensitive(self):
        """ShortName matching ignores case and surrounding whitespace.

        Test scenario:
            ``"  2T  "`` resolves the same as ``"2t"``.
        """
        assert style_for_parameter("  2T  ") == "temperature_2m", "lookup should be case/space-insensitive"

    def test_descriptive_name_passes_through(self):
        """A descriptive preset name is accepted directly.

        Test scenario:
            Passing an already-descriptive registered name returns it unchanged.
        """
        assert style_for_parameter("temperature_2m") == "temperature_2m", "descriptive names should pass through"

    @pytest.mark.parametrize("value", ["", "not-a-parameter", "xyz"])
    def test_unknown_returns_none(self, value):
        """An unknown or empty parameter returns ``None``.

        Args:
            value: An unmapped/empty parameter string.

        Test scenario:
            Callers can fall back to a default when no style matches.
        """
        assert style_for_parameter(value) is None, f"{value!r} should not resolve to a style"

    def test_every_mapped_name_is_a_registered_preset(self):
        """Every shortName in the map resolves to a live ``DATA_STYLES`` preset.

        Test scenario:
            The vendored map stays in sync with the shipped presets (no dangling
            descriptive names).
        """
        missing = [name for name in _SHORTNAME_TO_STYLE.values() if name not in DATA_STYLES]
        assert not missing, f"mapped names absent from DATA_STYLES: {missing}"
