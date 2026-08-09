"""Regression tests for the round-1 review findings on the style-presets branch."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
from cleopatra.styling.params import CellValues
from cleopatra.styling.colors import DATA_STYLES, convert_units, resolve_colormap, resolve_style_norm
from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph
from cleopatra.templates import publication_map


class TestReviewFixes:
    """One test per fixed review finding (H1, M1, L1, L3)."""

    def test_style_and_projection_composes_h1(self):
        """H1: `style` and `projection` compose (styled globe), not silently dropped.

        Test scenario:
            `publication_map(style=..., projection="globe")` reprojects the styled
            field and draws the globe boundary, rather than rendering a flat
            styled map. (Originally H1 only warned; A4 makes them compose.)
        """
        pytest.importorskip("pyproj", reason="globe needs the [tiles] extra")
        lon = np.linspace(-10.0, 10.0, 12)
        lat = np.linspace(50.0, 30.0, 10)
        data = np.random.default_rng(0).random((10, 12)) * 30.0
        fig, ax = publication_map(data, coords=(lon, lat), style="temperature_2m", projection="globe")
        assert len(ax.patches) >= 1, "styled globe should draw the boundary patch"
        plt.close("all")

    def test_radar_reflectivity_keeps_max_extend_m1(self):
        """M1: `radar_reflectivity` keeps `extend="max"` (spare over-range colour).

        Test scenario:
            The preset ships one more colour than bands, so `resolve_style_norm`
            does not downgrade the extend and the >75 dBZ arrow cap survives.
        """
        cfg = DATA_STYLES["radar_reflectivity"]["radar_reflectivity"]
        norm, _, _ = resolve_style_norm(np.linspace(0, 80, 50).reshape(5, 10), cfg)
        assert norm.extend == "max", f"radar extend should stay 'max', got {norm.extend!r}"

    def test_two_unknown_units_warns_l1(self):
        """L1: converting between two unrecognised units warns (contract).

        Test scenario:
            Both units unknown and different must still warn, not silently
            short-circuit on `None == None`.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = convert_units(np.array([1.0, 2.0]), "meters", "feet")
        assert len(caught) == 1, f"expected one warning for two unknown units, got {len(caught)}"
        assert result.tolist() == [1.0, 2.0], "data must be unchanged when no conversion applies"

    def test_cmap_none_passes_through_l3(self):
        """L3: `resolve_colormap(None)` returns None (matplotlib default preserved).

        Test scenario:
            A `None` cmap must pass through so the routed glyph seams keep
            matplotlib's default instead of raising `KeyError`.
        """
        assert resolve_colormap(None) is None, "None must pass through resolve_colormap"

    def test_scatter_cmap_none_renders_l3(self):
        """L3: a routed glyph seam still renders with `cmap=None`.

        Test scenario:
            `ScatterGlyph(..., cmap=None)` used to render on main; it must not
            regress to a `KeyError` after the resolver routing.
        """
        glyph = ScatterGlyph(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 0.0]),
            values=np.array([1.0, 2.0, 3.0]),
            cmap=None,
        )
        glyph.plot()
        assert glyph.fig is not None, "scatter with cmap=None should render"
        plt.close("all")

    def test_projection_with_overlay_warns_l5(self):
        """L5: `projection` + a cell-value overlay warns about misplacement.

        Test scenario:
            Overlays are drawn at raw grid indices under a projection, so the
            combination must warn.
        """
        lon = np.linspace(-10.0, 10.0, 8)
        lat = np.linspace(30.0, 50.0, 6)
        data = np.random.default_rng(0).random((6, 8))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ArrayGlyph(data, coords=(lon, lat), projection="flat").plot(
                cells=CellValues(show=True)
            )
        assert any("reprojected coordinates" in str(w.message) for w in caught), "projection+overlay must warn"
        plt.close("all")
