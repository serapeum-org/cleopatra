"""Tests for the Crameri terrain presets, NCL/MeteoSwiss tables, and the
hinge-faithful `_preset_cmap` loader (F1/F1b/F2).

Covers the palette colormap modes of `cleopatra.styling.colors._preset_cmap`
(`"linear"`, `"listed"`, `"perceptual"`), the registered terrain and NCL
presets, and the fixed `topography` hinge registration.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from cleopatra.styling.colors import DATA_STYLES, _preset_cmap, apply_data_style

#: A three-colour palette whose middle control point (pure green) sits at index
#: 0.5, used to probe where each interpolation mode places the hinge.
PALETTE = ["#000000", "#00ff00", "#ffffff"]


def _fraction_of(cmap, target_rgb, samples=1024):
    """Return the 0-1 fraction where `cmap` is closest to `target_rgb`.

    Args:
        cmap: A matplotlib colormap.
        target_rgb: An ``(r, g, b)`` triple in 0-1.
        samples: Number of samples across the ramp. Defaults to 1024.

    Returns:
        float: The fraction of the closest-matching sample.
    """
    xs = np.linspace(0.0, 1.0, samples)
    cols = cmap(xs)[:, :3]
    return float(xs[np.argmin(((cols - np.array(target_rgb)) ** 2).sum(axis=1))])


class TestAssetCmap:
    """Tests for the `_preset_cmap` colormap-mode dispatch."""

    def test_linear_preserves_hinge_at_half(self):
        """`colormap="linear"` keeps the middle control point at fraction 0.5.

        Test scenario:
            A `from_list` ramp spaces control points evenly by index, so the
            green midpoint of a 3-colour palette stays at 0.5 -- the property
            hinge maps rely on.
        """
        cmap = _preset_cmap("t", PALETTE, "linear")
        assert isinstance(cmap, LinearSegmentedColormap), "linear must be a segmented map"
        assert abs(_fraction_of(cmap, (0.0, 1.0, 0.0)) - 0.5) < 0.02, "hinge should stay at 0.5"

    def test_perceptual_default_drifts_hinge(self):
        """`colormap="perceptual"` reparameterises, moving the midpoint off 0.5.

        Test scenario:
            CIELAB arc-length spacing is uneven for this palette, so the green
            midpoint lands away from 0.5 -- demonstrating why hinge maps need
            `"linear"` instead.
        """
        cmap = _preset_cmap("t", PALETTE, "perceptual")
        assert isinstance(cmap, LinearSegmentedColormap), "perceptual returns a segmented map"
        assert abs(_fraction_of(cmap, (0.0, 1.0, 0.0)) - 0.5) > 0.02, "perceptual should drift the midpoint"

    def test_listed_is_discrete_with_one_band_per_colour(self):
        """`colormap="listed"` yields a `ListedColormap` with `N == len(palette)`.

        Test scenario:
            A stepped colour table must stay discrete (one flat band per stored
            colour), not interpolate to a smooth ramp.
        """
        cmap = _preset_cmap("t", PALETTE, "listed")
        assert isinstance(cmap, ListedColormap), "listed must be a discrete map"
        assert cmap.N == len(PALETTE), f"expected {len(PALETTE)} bands, got {cmap.N}"

    def test_named_passes_the_name_through(self):
        """`colormap="named"` returns the name string unchanged (resolved at draw).

        Test scenario:
            A named colormap is not built up front; the loader keeps the string so
            the render path resolves it (matplotlib or namespaced).
        """
        assert _preset_cmap("t", "Spectral_r", "named") == "Spectral_r"

    @pytest.mark.parametrize("colormap", ["linear", "listed", "perceptual"])
    def test_single_colour_palette_raises(self, colormap):
        """A <2-colour palette raises under every palette colormap mode.

        Args:
            colormap: The palette colormap mode.

        Test scenario:
            The uniform guard lets `_load_presets` skip a degenerate record the
            same way for all palette modes.
        """
        with pytest.raises(ValueError, match="at least two colours"):
            _preset_cmap("t", ["#000000"], colormap)


class TestTopographyHingeFix:
    """Tests for the fixed `topography` preset registration (F1b)."""

    def test_topography_is_linear_and_centered(self):
        """`topography` is a linear (hinge-faithful) map centred at zero.

        Test scenario:
            The fix re-registers the cmocean topography palette with
            `interp="linear"` so it is a `LinearSegmentedColormap` and keeps its
            `center=0`.
        """
        layer = DATA_STYLES["topography"]["topography"]
        assert isinstance(layer["cmap"], LinearSegmentedColormap), "topography must be a segmented map"
        assert layer["center"] == 0.0, "topography must stay centred at sea level"

    def test_topography_sea_level_near_half(self):
        """The shallow-water / low-land break sits near cmap fraction 0.5.

        Test scenario:
            The first land colour (`#0f2915`) resolves close to 0.5 under the
            linear loader, so `center=0` lines sea level up with the hinge
            (perceptual loading drifted it to ~0.65).
        """
        cmap = DATA_STYLES["topography"]["topography"]["cmap"]
        dark_land = (0x0F / 255, 0x29 / 255, 0x15 / 255)
        assert abs(_fraction_of(cmap, dark_land) - 0.5) < 0.03, "sea-level hinge should sit near 0.5"


class TestTerrainPresets:
    """Tests for the vendored Crameri terrain presets (F1)."""

    @pytest.mark.parametrize("key", ["elevation_oleron", "elevation_bukavu", "elevation_fes"])
    def test_terrain_preset_registered_linear_centered(self, key):
        """Each terrain preset is a linear map centred at zero.

        Args:
            key: The terrain preset name.

        Test scenario:
            Crameri hypsometric maps are registered hinge-faithful
            (`LinearSegmentedColormap`) and symmetric about sea level
            (`center=0`).
        """
        layer = DATA_STYLES[key][key]
        assert isinstance(layer["cmap"], LinearSegmentedColormap), f"{key} must be a segmented map"
        assert layer["center"] == 0.0, f"{key} must be centred at zero"

    @pytest.mark.parametrize("key", ["elevation_oleron", "elevation_bukavu", "elevation_fes"])
    def test_terrain_preset_renders(self, key):
        """Each terrain preset draws through `apply_data_style` without error.

        Args:
            key: The terrain preset name.

        Test scenario:
            A signed elevation field renders end-to-end under the preset.
        """
        data = np.linspace(-4000, 4000, 600).reshape(20, 30)
        fig, ax = plt.subplots()
        apply_data_style(ax, {key: data}, style=key)
        assert ax.images, f"{key} should draw a raster"
        plt.close(fig)


class TestNCLPresets:
    """Tests for the vendored NCL/MeteoSwiss stepped tables (F2)."""

    @pytest.mark.parametrize(
        "key",
        [
            "precipitation_steps",
            "precipitation_anomaly",
            "temperature_steps",
            "temperature_anomaly",
            "sunshine_hours",
            "hot_cold",
        ],
    )
    def test_ncl_preset_is_listed(self, key):
        """Each NCL preset is a discrete `ListedColormap`.

        Args:
            key: The NCL preset name.

        Test scenario:
            The stepped MeteoSwiss tables load as banded (`ListedColormap`),
            not smooth ramps, so the operational look is preserved.
        """
        cmap = DATA_STYLES[key][key]["cmap"]
        assert isinstance(cmap, ListedColormap), f"{key} must be a discrete banded map"
        assert cmap.N >= 2, f"{key} should have at least two bands, got {cmap.N}"

    @pytest.mark.parametrize("key", ["precipitation_anomaly", "temperature_anomaly", "hot_cold"])
    def test_anomaly_presets_are_centered(self, key):
        """The diverging anomaly tables carry `center=0`.

        Args:
            key: The anomaly preset name.

        Test scenario:
            Difference tables render symmetric about zero.
        """
        assert DATA_STYLES[key][key]["center"] == 0.0, f"{key} must be centred at zero"

    @pytest.mark.parametrize("key", ["precipitation_steps", "temperature_anomaly", "hot_cold"])
    def test_ncl_preset_renders(self, key):
        """Each NCL preset draws through `apply_data_style` without error.

        Args:
            key: The NCL preset name.

        Test scenario:
            A signed field renders end-to-end under the stepped table.
        """
        data = np.random.default_rng(0).standard_normal((20, 30)) * 20
        fig, ax = plt.subplots()
        apply_data_style(ax, {key: data}, style=key)
        assert ax.images, f"{key} should draw a raster"
        plt.close(fig)
