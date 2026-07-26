"""Tests for `cleopatra.palettes` -- the unified Palette record and registry."""
import pytest
from matplotlib.colors import Colormap, ListedColormap

from cleopatra.palettes import (
    PALETTES,
    Palette,
    PaletteKind,
    available_palettes,
    get_palette,
    register,
)


class TestPaletteKind:
    """The PaletteKind string enum."""

    def test_string_equality(self):
        """Members compare equal to their string value."""
        assert PaletteKind.DIVERGING == "diverging"

    def test_case_insensitive_construction(self):
        """Construction is case-insensitive and hyphen-tolerant."""
        assert PaletteKind("Qualitative") is PaletteKind.QUALITATIVE
        assert PaletteKind("SEQUENTIAL") is PaletteKind.SEQUENTIAL

    @pytest.mark.parametrize("bad", ["nonsense", 123])
    def test_invalid_value_raises(self, bad):
        """An unknown string or a non-string value raises ValueError."""
        with pytest.raises(ValueError):
            PaletteKind(bad)


class TestPalette:
    """The Palette record."""

    def test_coerces_kind_and_colors(self):
        """A string kind and list of colours are coerced to the canonical types."""
        p = Palette("p", "sequential", ["#ffffff", "#000000"])
        assert p.kind is PaletteKind.SEQUENTIAL
        assert isinstance(p.colors, tuple)

    def test_sequential_builds_interpolated_colormap(self):
        """A sequential palette becomes a named perceptually-interpolated Colormap."""
        cmap = Palette("s", "sequential", ("#ffffff", "#004cff")).to_colormap()
        assert isinstance(cmap, Colormap)
        assert cmap.name == "s"

    def test_qualitative_keeps_exact_swatches(self):
        """A qualitative palette becomes a ListedColormap of its exact colours."""
        cols = ("#ff0000", "#00ff00", "#0000ff")
        cmap = Palette("q", "qualitative", cols).to_colormap()
        assert isinstance(cmap, ListedColormap)
        assert cmap.N == 3

    def test_default_source(self):
        """Source defaults to 'cleopatra'."""
        assert Palette("p", "cyclic", ("#000", "#fff")).source == "cleopatra"


class TestRegistry:
    """register / get_palette / available_palettes."""

    @pytest.fixture(autouse=True)
    def _isolate(self):
        """Snapshot and restore the global registry around each test."""
        saved = dict(PALETTES)
        yield
        PALETTES.clear()
        PALETTES.update(saved)

    def test_register_and_get(self):
        """A registered palette is retrievable by name."""
        register(Palette("temp", "diverging", ("#762a83", "#f4f4f4", "#1b7837")))
        assert get_palette("temp").kind is PaletteKind.DIVERGING

    def test_get_unknown_raises(self):
        """Looking up an unregistered name raises KeyError."""
        with pytest.raises(KeyError, match="unknown palette"):
            get_palette("does_not_exist")

    def test_available_filters_by_kind(self):
        """available_palettes(kind) returns only that kind, sorted."""
        register(Palette("q_one", "qualitative", ("#000", "#fff")))
        got = available_palettes("qualitative")
        assert "q_one" in got and got == sorted(got)
        assert "q_one" not in available_palettes("sequential")


class TestBuiltinsRegistered:
    """The built-in haze/cams/flame families register at import."""

    @pytest.mark.parametrize(
        "name", ["haze_dust", "haze_organic_matter", "cams_aod_blue_red", "flame_white_hot"]
    )
    def test_present_as_sequential(self, name):
        """Each built-in colour family is registered as a sequential palette."""
        assert get_palette(name).kind is PaletteKind.SEQUENTIAL

    def test_cams_records_magics_provenance(self):
        """The CAMS palettes record their ECMWF/Magics provenance in source."""
        assert get_palette("cams_aod_blue_red").source == "ecmwf-magics"
