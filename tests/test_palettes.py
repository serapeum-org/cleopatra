"""Tests for `cleopatra.palettes` -- the unified Palette record and registry."""
import numpy as np
import pytest
from matplotlib.colors import (
    BoundaryNorm,
    CenteredNorm,
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
)

from cleopatra.perceptual import srgb_to_lab

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


class TestDefaultNorm:
    """Palette.default_norm -- the kind-driven norm."""

    def test_sequential_is_linear_normalize(self):
        """A sequential palette yields a plain linear Normalize over the bounds."""
        norm = Palette("s", "sequential", ("#fff", "#000")).default_norm(vmin=2, vmax=8)
        assert type(norm) is Normalize
        assert (norm.vmin, norm.vmax) == (2, 8)

    def test_sequential_autoranges_from_data(self):
        """With no explicit bounds, sequential auto-ranges from the finite data range."""
        data = np.array([[1.0, np.nan], [3.0, 5.0]])
        norm = Palette("s", "sequential", ("#fff", "#000")).default_norm(data)
        assert (norm.vmin, norm.vmax) == (1.0, 5.0)

    def test_diverging_is_symmetric_centered(self):
        """A diverging palette yields a CenteredNorm symmetric about 0 by default."""
        norm = Palette("d", "diverging", ("#00f", "#fff", "#f00")).default_norm(vmin=-5, vmax=8)
        assert isinstance(norm, CenteredNorm)
        assert norm.vcenter == 0.0
        assert norm.halfrange == 8.0  # max(|-5|, |8|)

    def test_diverging_honours_center(self):
        """An explicit center shifts the diverging norm's midpoint and halfrange."""
        norm = Palette("d", "diverging", ("#00f", "#fff", "#f00")).default_norm(
            vmin=10, vmax=30, center=20
        )
        assert norm.vcenter == 20.0
        assert norm.halfrange == 10.0

    def test_diverging_without_bounds_autoscales(self):
        """With no bounds or data, diverging returns a CenteredNorm that autoscales at draw."""
        norm = Palette("d", "diverging", ("#00f", "#fff", "#f00")).default_norm()
        assert isinstance(norm, CenteredNorm)
        assert norm.vcenter == 0.0 and norm.halfrange is None

    def test_qualitative_is_indexed_boundary_norm(self):
        """A qualitative palette yields a BoundaryNorm mapping class index k to swatch k."""
        norm = Palette("q", "qualitative", ("#f00", "#0f0", "#00f")).default_norm()
        assert isinstance(norm, BoundaryNorm)
        assert norm.Ncmap == 3  # one colour slot per class
        assert int(norm(0)) == 0 and int(norm(2)) == 2

    def test_cyclic_is_linear_normalize(self):
        """A cyclic palette uses a linear Normalize (the wrapping lives in the colormap)."""
        norm = Palette("c", "cyclic", ("#f00", "#0f0", "#f00")).default_norm(vmin=0, vmax=360)
        assert type(norm) is Normalize

    def test_explicit_bounds_ignore_data(self):
        """When both vmin and vmax are given, the data range is not consulted."""
        data = np.array([-999.0, 999.0])
        norm = Palette("s", "sequential", ("#fff", "#000")).default_norm(data, vmin=0, vmax=1)
        assert (norm.vmin, norm.vmax) == (0, 1)

    def test_all_nan_data_leaves_bounds_unset(self):
        """All-NaN data yields no finite range, so bounds stay None (autoscale at draw)."""
        norm = Palette("s", "sequential", ("#fff", "#000")).default_norm(np.full(4, np.nan))
        assert norm.vmin is None and norm.vmax is None


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


class TestCuratedPalettes:
    """The curated, generated palettes registered at import (not vendored)."""

    @pytest.mark.parametrize(
        "name, kind",
        [
            ("diverging_blue_red", PaletteKind.DIVERGING),
            ("diverging_purple_green", PaletteKind.DIVERGING),
            ("diverging_brown_teal", PaletteKind.DIVERGING),
            ("category12", PaletteKind.QUALITATIVE),
            ("category20", PaletteKind.QUALITATIVE),
        ],
    )
    def test_registered_with_expected_kind(self, name, kind):
        """Each curated palette is registered under its expected kind."""
        assert get_palette(name).kind is kind

    def test_diverging_builds_continuous_with_light_centre(self):
        """A curated diverging palette builds a continuous map whose centre is lighter than its ends."""
        cmap = get_palette("diverging_blue_red").to_colormap()
        assert isinstance(cmap, LinearSegmentedColormap)
        l_mid = srgb_to_lab(cmap(0.5)[:3])[0]
        l_lo = srgb_to_lab(cmap(0.0)[:3])[0]
        l_hi = srgb_to_lab(cmap(1.0)[:3])[0]
        assert l_mid > l_lo and l_mid > l_hi

    @pytest.mark.parametrize("name, n", [("category12", 12), ("category20", 20)])
    def test_category_palettes_are_listed_and_distinct(self, name, n):
        """The categorical palettes hold n distinct swatches and build an n-colour ListedColormap."""
        pal = get_palette(name)
        assert len(pal.colors) == n == len(set(pal.colors))
        cmap = pal.to_colormap()
        assert isinstance(cmap, ListedColormap) and cmap.N == n

    def test_category_colours_are_well_separated(self):
        """category12's minimum pairwise CIELAB distance clears a distinguishability floor."""
        lab = srgb_to_lab(
            np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]
                      for c in get_palette("category12").colors])
        )
        dists = [np.sqrt(((lab[i] - lab[j]) ** 2).sum())
                 for i in range(len(lab)) for j in range(i + 1, len(lab))]
        assert min(dists) > 25.0
