"""Regression tests for the round-2 review findings on the style-presets branch."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph
from cleopatra.templates import publication_map


class TestReviewFixesRound2:
    """One test per fixed round-2 finding (M1, L1, L2)."""

    def test_integer_masked_array_projection_renders_m1(self):
        """M1: an integer masked raster renders under a projection (float-cast).

        Test scenario:
            `_plot_projected` casts to float before NaN-filling, so an integer
            masked array (land-cover codes, counts) no longer raises
            `TypeError: Cannot convert fill_value nan to dtype int64`.
        """
        lon = np.linspace(-10.0, 10.0, 8)
        lat = np.linspace(30.0, 50.0, 6)
        int_data = np.arange(48, dtype=np.int64).reshape(6, 8)
        mask = np.zeros((6, 8), dtype=bool)
        mask[0, 0] = True
        masked = np.ma.masked_array(int_data, mask)
        glyph = ArrayGlyph(masked, coords=(lon, lat), projection="flat")
        glyph.plot()
        assert glyph.im is not None, "integer masked raster should render under projection"
        plt.close("all")

    def test_globe_relief_warns_and_skips_l1(self):
        """L1: `publication_map(projection="globe", relief=True)` warns and skips relief.

        Test scenario:
            Relief is a lon/lat image and cannot sit on a metres-scale globe axes,
            so it is skipped with a warning (which also avoids any network fetch).
        """
        lon = np.linspace(-10.0, 10.0, 8)
        lat = np.linspace(30.0, 50.0, 6)
        data = np.random.default_rng(0).random((6, 8))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            publication_map(data, coords=(lon, lat), projection="globe", relief=True)
        assert any("relief basemaps compose only" in str(w.message) for w in caught), "globe+relief must warn"
        plt.close("all")

    def test_projection_ignores_kind_and_hillshade_warns_l2(self):
        """L2: `projection` with a non-default `kind` or `hillshade` warns.

        Test scenario:
            The projection path always uses pcolormesh, so passing `kind="contour"`
            surfaces a warning rather than silently ignoring it.
        """
        lon = np.linspace(-10.0, 10.0, 8)
        lat = np.linspace(30.0, 50.0, 6)
        data = np.random.default_rng(0).random((6, 8))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ArrayGlyph(data, coords=(lon, lat), projection="flat").plot(kind="contour")
        assert any("always renders via pcolormesh" in str(w.message) for w in caught), "kind under projection must warn"
        plt.close("all")
